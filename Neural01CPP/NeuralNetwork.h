#pragma once

#include <barrier>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

using namespace std;

class NeuralNetwork {
public:
    // One fully connected hidden layer; sizes are independent of the task.
    NeuralNetwork(size_t inputs, size_t hidden, size_t outputs, size_t workers = 1, uint32_t seed = 42);
    ~NeuralNetwork();
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    void train(const vector<double>& input,
               const vector<double>& target, double learningRate);
    vector<double> predict(const vector<double>& input);

private:
    static size_t checkedWorkers(size_t inputs, size_t hidden,
                                      size_t outputs, size_t workers);
    static double sigmoid(double value);
    static double derivative(double activation);
    void validateInput(const vector<double>& input) const;
    void workerLoop(stop_token stop, size_t worker);
    void trainingStep(size_t worker);

    const size_t inputs_, hidden_, outputs_, workerCount_;
    // Flat row-major matrices: weight[neuron * previousLayerSize + input].
    vector<double> hiddenWeights_, hiddenBias_, outputWeights_, outputBias_;
    // Reused scratch space; no allocation is needed inside a training step.
    vector<double> hiddenValues_, outputValues_, hiddenDelta_, outputDelta_;

    // API mutex prevents two callers from starting overlapping operations.
    mutex apiMutex_;
    // Job mutex + condition variable publish work and sleep idle workers.
    mutex jobMutex_;
    condition_variable_any jobReady_;
    size_t generation_ = 0;
    const vector<double>* input_ = nullptr;
    const vector<double>* target_ = nullptr;
    double learningRate_ = 0;
    // All workers participate in each stage barrier, even with no assigned rows.
    barrier<> stageBarrier_;
    // Worker zero signals the calling thread when the entire step is finished.
    binary_semaphore completed_{0};
    // Declared last: workers must stop before their shared state is destroyed.
    vector<jthread> workers_;
};
