#include "NeuralNetwork.h"
#include "Tasks.h"

#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>

using namespace std;

size_t positiveNumber(string_view text) {
    size_t value = 0;
    const auto result = from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != errc{} || result.ptr != text.data() + text.size() || value == 0) {
        throw invalid_argument("Arguments must be positive integers");
    }
    return value;
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 3) {
            throw invalid_argument("Usage: ./neural01 [workers] [training-steps]");
        }
        // 1. Application configuration. Two workers demonstrate synchronization;
        // use one to measure the sequential baseline. XOR is too small for speedup.
        const size_t workers = argc > 1 ? positiveNumber(argv[1]) : 2;
        const size_t steps = argc > 2 ? positiveNumber(argv[2]) : 100000;
        const double learningRate = 0.1;
        const auto samples = xorSamples();
        NeuralNetwork network(2, 4, 1, workers, 42);

        // 2. Randomly sample with replacement, as in the C# training loop.
        // Initialization and sampling have separate fixed seeds for repeatability.
        mt19937 random(0);
        uniform_int_distribution<size_t> choose(0, samples.size() - 1);
        cout << "XOR: 2 -> 4 -> 1; workers=" << workers << "; steps=" << steps << '\n';
        const auto start = chrono::steady_clock::now();
        for (size_t step = 0; step < steps; ++step) {
            const auto& sample = samples[choose(random)];
            // 3. This returns only after forward propagation, backpropagation,
            // and every parameter update finish. The next example sees new weights.
            network.train(sample.input, sample.target, learningRate);
        }
        const double seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();

        // 4. Evaluate without learning. Thresholding belongs to this XOR task,
        // not to the engine: digit classification would instead use argmax.
        double loss = 0;
        cout << fixed << setprecision(6);
        for (const auto& sample : samples) {
            const double output = network.predict(sample.input)[0];
            const double error = sample.target[0] - output;
            loss += error * error;
            cout << sample.input[0] << " XOR " << sample.input[1] << " -> "
                 << output << " => " << (output > 0.5 ? 1 : 0) << '\n';
        }
        cout << "Mean squared error: " << loss / samples.size()
             << "\nTraining seconds: " << seconds << '\n';
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
