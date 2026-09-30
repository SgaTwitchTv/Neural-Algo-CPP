#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace std;

// Task-specific data stays outside the engine. A future image loader can return
// these same samples: 784 normalized pixels and a ten-element one-hot target.
struct Sample {
    vector<double> input;
    vector<double> target;
};

inline vector<Sample> xorSamples() {
    return {{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};
}

inline vector<double> oneHot(size_t label, size_t classes) {
    if (label >= classes) {
        throw invalid_argument("Label is outside class range");
    }
    vector<double> target(classes, 0.0);
    target[label] = 1.0;
    return target;
}
