#include "NeuralNetwork.h"
#include "Tasks.h"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;

void require(bool condition, const char* message) {
    if (!condition) {
        throw runtime_error(message);
    }
}

int main() {
    try {
        // Compare multiple output neurons and uneven row assignments. More
        // workers than outputs also checks that idle workers reach every barrier.
        NeuralNetwork sequential(3, 7, 2, 1, 17);
        NeuralNetwork parallel(3, 7, 2, 4, 17);
        const vector<double> input{0.2, 0.8, -0.3}, target{1, 0};
        for (int i = 0; i < 300; ++i) {
            sequential.train(input, target, 0.1);
            parallel.train(input, target, 0.1);
            require(sequential.predict(input) == parallel.predict(input),
                    "Parallel execution diverged from sequential execution");
        }

        // Independently calculated first update for a 1-1-1 network checks the
        // gradient sign, bias updates, and use of old output weights in backprop.
        mt19937 initialization(9);
        uniform_real_distribution<double> distribution(-1, 1);
        double w = distribution(initialization), b = distribution(initialization);
        double v = distribution(initialization), c = distribution(initialization);
        auto sigmoid = [](double z) { return 1.0 / (1.0 + exp(-z)); };
        double h = sigmoid(b + w * 0.6), o = sigmoid(c + v * h);
        double od = (1 - o) * o * (1 - o), hd = od * v * h * (1 - h);
        v += 0.2 * od * h; c += 0.2 * od;
        w += 0.2 * hd * 0.6; b += 0.2 * hd;
        NeuralNetwork tiny(1, 1, 1, 3, 9);
        tiny.train({0.6}, {1}, 0.2);
        require(abs(tiny.predict({0.6})[0] - sigmoid(c + v * sigmoid(b + w * 0.6))) < 1e-14,
                "One-step update disagrees with independent calculation");

        // End-to-end learning verifies that all four XOR cases are learned.
        NeuralNetwork xorNetwork(2, 4, 1, 1, 42);
        auto samples = xorSamples();
        mt19937 random(0);
        uniform_int_distribution<size_t> choose(0, 3);
        for (int i = 0; i < 100000; ++i) {
            const auto& sample = samples[choose(random)];
            xorNetwork.train(sample.input, sample.target, 0.1);
        }
        for (const auto& sample : samples) {
            require(abs(xorNetwork.predict(sample.input)[0] - sample.target[0]) < 0.1,
                    "XOR did not converge");
        }

        bool rejected = false;
        try { parallel.train({1}, target, 0.1); }
        catch (const invalid_argument&) { rejected = true; }
        require(rejected, "Invalid dimensions were accepted");
        // A rejected call must not strand workers or poison the next job.
        parallel.train(input, target, 0.1);
        cout << "PASS: numerical update, thread parity, XOR convergence, validation and shutdown\n";
    } catch (const exception& error) {
        cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
