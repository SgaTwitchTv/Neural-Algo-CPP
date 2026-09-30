# Neural01CPP

A C++20 implementation of the original single-hidden-layer C# network:
dense connections, sigmoid in both layers, random weights and biases in [-1,1),
and gradient descent on half squared error after each sampled example.
It uses standard C++ threads and synchronization, backed by OS facilities.

## Build and run

```sh
cd Neural01CPP
make
./neural01                 # Two workers, 100,000 sampled training steps
./neural01 1 100000        # Sequential baseline
./neural01 4 100000        # Four persistent worker threads
make test
make sanitize             # AddressSanitizer and UndefinedBehaviorSanitizer
```

Requires g++ with C++20 support and pthreads. Steps are individual examples,
not dataset epochs. XOR is deliberately tiny: multiple threads will generally
be slower. Compare timing with identical step counts; no speedup is promised.

## File map

- `main.cpp`: task selection, configuration, sampling, training and evaluation.
- `NeuralNetwork.h`: reusable engine interface and synchronization state.
- `NeuralNetwork.cpp`: commented mathematics, worker lifecycle and stage execution.
- `Tasks.h`: sample representation, XOR data and one-hot helper.
- `tests.cpp`: numerical reference, sequential/parallel equivalence and learning checks.

## Synchronization

The API mutex serializes calls to train/predict on one network. A separate job
mutex and stop-aware condition variable publish an example and wake workers.
Each worker owns neuron rows by its index modulo the worker count.

1. Compute hidden activations; barrier.
2. Compute output activations and output deltas; barrier.
3. Compute hidden deltas using unchanged output weights; barrier.
4. Update disjoint parameter rows; barrier.
5. Worker zero releases a binary semaphore so train() can return.

This preserves one-example-at-a-time training. It does not run different
examples concurrently or use mini-batches. Reductions within each neuron keep
their original order, allowing exact comparisons between worker counts in the
same build. One-worker mode executes directly without a background thread.
Workers sleep between jobs and are stopped and joined during destruction.
Inputs must remain unmodified during calls; destroy the network only after all
external callers finish. Prediction currently runs sequentially under the API lock.

## Extending to another task

Create a loader that supplies `Sample` objects and change main's layer sizes.
For digits, construct `NeuralNetwork(784, 128, 10, workers, seed)`, normalize
28x28 pixels to [0,1], and use `oneHot(label, 10)` for the target. Select the
largest output for classification. Outputs are sigmoid scores, not normalized
class probabilities. No MNIST loader or image UI is implemented here yet.

The engine has no XOR threshold, file format, UI or label assumptions beyond
finite inputs and sigmoid targets in [0,1]. It supports exactly one hidden
layer; arbitrary depth, other activations and losses require engine changes.
Seeds make this C++ implementation reproducible within a given standard-library
implementation; C# and C++ random sequences are not identical.
