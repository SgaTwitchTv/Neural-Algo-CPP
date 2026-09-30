#include "NeuralNetwork.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using namespace std;

size_t NeuralNetwork::checkedWorkers(size_t inputs, size_t hidden, size_t outputs, size_t workers) {
    if (!inputs || !hidden || !outputs || !workers)
    {
        throw invalid_argument("Layer sizes and worker count must be positive");
    }
    if (hidden > numeric_limits<size_t>::max() / inputs || outputs > numeric_limits<size_t>::max() / hidden)
    {
        throw invalid_argument("Weight matrix size overflow");
    }
    if (workers > static_cast<size_t>(barrier<>::max()))
    {
        throw invalid_argument("Too many workers for the barrier");
    }

    return workers;
}

NeuralNetwork::NeuralNetwork(size_t inputs, size_t hidden, size_t outputs, size_t workers, uint32_t seed)
    : inputs_(inputs), hidden_(hidden), outputs_(outputs),
      workerCount_(checkedWorkers(inputs, hidden, outputs, workers)),
      hiddenWeights_(hidden * inputs), hiddenBias_(hidden),
      outputWeights_(outputs * hidden), outputBias_(outputs),
      hiddenValues_(hidden), outputValues_(outputs),
      hiddenDelta_(hidden), outputDelta_(outputs),
      stageBarrier_(static_cast<ptrdiff_t>(workerCount_)) {
    // 1. Break symmetry by initializing weights AND biases in [-1, 1).
    // A seed makes comparisons between worker counts reproducible.
    mt19937 random(seed);
    uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (auto* values : {&hiddenWeights_, &hiddenBias_, &outputWeights_, &outputBias_}) {
        for (double& value : *values) {
            value = distribution(random);
        }
    }

    // 2. Create workers once, not once per example. With one worker, train()
    // executes directly on the caller as a sequential performance baseline.
    if (workerCount_ > 1)
    {
        workers_.reserve(workerCount_);
        for (size_t id = 0; id < workerCount_; ++id)
        {
            workers_.emplace_back([this, id](stop_token stop) { workerLoop(stop, id); });
        }
    }
}

NeuralNetwork::~NeuralNetwork() {
    // Stop-aware waits wake on request_stop(). Join before destroying shared data.
    // As with ordinary C++ objects, callers must finish using it before destruction.
    for (auto& worker : workers_) {
        worker.request_stop();
    }
    for (auto& worker : workers_) {
        worker.join();
    }
}

double NeuralNetwork::sigmoid(double value) {
    // Equivalent to 1/(1+exp(-value)), with no exponential overflow.
    if (value >= 0) {
        return 1.0 / (1.0 + exp(-value));
    }
    const double exponential = exp(value);
    return exponential / (1.0 + exponential);
}

double NeuralNetwork::derivative(double activation) {
    // This receives sigmoid(z), NOT the pre-activation weighted sum z.
    return activation * (1.0 - activation);
}

void NeuralNetwork::validateInput(const vector<double>& input) const {
    if (input.size() != inputs_ ||
        !all_of(input.begin(), input.end(), [](double x) { return isfinite(x); })) {
        throw invalid_argument("Input must have the configured size and finite values");
    }
}

void NeuralNetwork::train(const vector<double>& input,
                          const vector<double>& target, double learningRate) {
    lock_guard apiLock(apiMutex_);
    validateInput(input);
    if (target.size() != outputs_ ||
        !all_of(target.begin(), target.end(), [](double x) {
            return isfinite(x) && x >= 0 && x <= 1;
        }) || !isfinite(learningRate) || learningRate <= 0) {
        throw invalid_argument("Targets must be in [0,1]; learning rate must be positive and finite");
    }

    // 3. Publish one example. train() waits for completion, so these references
    // remain valid. The caller must not modify input/target during this call.
    {
        lock_guard jobLock(jobMutex_);
        input_ = &input;
        target_ = &target;
        learningRate_ = learningRate;
        ++generation_;
    }
    if (workerCount_ == 1) {
        trainingStep(0);
    } else {
        jobReady_.notify_all();
        completed_.acquire(); // Sleep until worker zero signals completion.
    }
}

void NeuralNetwork::workerLoop(stop_token stop, size_t worker) {
    size_t seenGeneration = 0;
    while (true) {
        {
            unique_lock jobLock(jobMutex_);
            // The predicate handles spurious wakeups and notifications arriving
            // before a worker starts waiting. The mutex publishes the job data.
            if (!jobReady_.wait(jobLock, stop, [&] { return generation_ != seenGeneration; })) {
                return;
            }
            seenGeneration = generation_;
        }
        trainingStep(worker);
        // The final barrier guarantees every worker has finished accessing the
        // example before the caller can return or publish the next example.
        if (worker == 0) {
            completed_.release();
        }
    }
}

void NeuralNetwork::trainingStep(size_t worker) {
    // Each worker owns rows worker, worker+workerCount, ... . No two workers
    // write the same neuron, bias, delta, or weight row in a given stage.

    // 4. Input -> hidden: h[i] = sigmoid(b[i] + sum_j W[i,j]*x[j]).
    for (size_t i = worker; i < hidden_; i += workerCount_) {
        double sum = hiddenBias_[i];
        for (size_t j = 0; j < inputs_; ++j) {
            sum += hiddenWeights_[i * inputs_ + j] * (*input_)[j];
        }
        hiddenValues_[i] = sigmoid(sum);
    }
    stageBarrier_.arrive_and_wait(); // Every output needs all hidden activations.

    // 5. Hidden -> output, then calculate each output's adjustment signal.
    // delta = (target-output)*output*(1-output), for half squared-error loss.
    for (size_t i = worker; i < outputs_; i += workerCount_) {
        double sum = outputBias_[i];
        for (size_t j = 0; j < hidden_; ++j) {
            sum += outputWeights_[i * hidden_ + j] * hiddenValues_[j];
        }
        outputValues_[i] = sigmoid(sum);
        outputDelta_[i] = ((*target_)[i] - outputValues_[i]) * derivative(outputValues_[i]);
    }
    stageBarrier_.arrive_and_wait(); // Hidden deltas need all output deltas.

    // 6. Backpropagate through the ORIGINAL output weights (chain rule).
    for (size_t i = worker; i < hidden_; i += workerCount_) {
        double error = 0;
        for (size_t j = 0; j < outputs_; ++j) {
            error += outputDelta_[j] * outputWeights_[j * hidden_ + i];
        }
        hiddenDelta_[i] = error * derivative(hiddenValues_[i]);
    }
    stageBarrier_.arrive_and_wait(); // No weight may change before this point.

    // 7. Gradient descent: += is correct because delta uses target-output.
    // Output and hidden matrices are disjoint, so both can be updated now.
    for (size_t i = worker; i < outputs_; i += workerCount_) {
        for (size_t j = 0; j < hidden_; ++j) {
            outputWeights_[i * hidden_ + j] += learningRate_ * outputDelta_[i] * hiddenValues_[j];
        }
        outputBias_[i] += learningRate_ * outputDelta_[i];
    }
    for (size_t i = worker; i < hidden_; i += workerCount_) {
        for (size_t j = 0; j < inputs_; ++j) {
            hiddenWeights_[i * inputs_ + j] += learningRate_ * hiddenDelta_[i] * (*input_)[j];
        }
        hiddenBias_[i] += learningRate_ * hiddenDelta_[i];
    }
    stageBarrier_.arrive_and_wait(); // Finish this example before starting another.
}

vector<double> NeuralNetwork::predict(const vector<double>& input) {
    lock_guard apiLock(apiMutex_); // Prediction must not read partially updated weights.
    validateInput(input);
    // Inference needs only a forward pass; local buffers leave training scratch alone.
    vector<double> hidden(hidden_), output(outputs_);
    for (size_t i = 0; i < hidden_; ++i) {
        double sum = hiddenBias_[i];
        for (size_t j = 0; j < inputs_; ++j) {
            sum += hiddenWeights_[i * inputs_ + j] * input[j];
        }
        hidden[i] = sigmoid(sum);
    }
    for (size_t i = 0; i < outputs_; ++i) {
        double sum = outputBias_[i];
        for (size_t j = 0; j < hidden_; ++j) {
            sum += outputWeights_[i * hidden_ + j] * hidden[j];
        }
        output[i] = sigmoid(sum);
    }
    return output;
}
