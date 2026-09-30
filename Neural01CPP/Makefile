CXX = g++
CPPFLAGS =
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Wpedantic
LDFLAGS =
LDLIBS = -pthread

.PHONY: all test sanitize clean
all: neural01

neural01: main.cpp NeuralNetwork.cpp NeuralNetwork.h Tasks.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp NeuralNetwork.cpp $(LDFLAGS) $(LDLIBS) -o $@

tests: tests.cpp NeuralNetwork.cpp NeuralNetwork.h Tasks.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests.cpp NeuralNetwork.cpp $(LDFLAGS) $(LDLIBS) -o $@

test: tests
	./tests

sanitize:
	$(CXX) $(CPPFLAGS) -std=c++20 -O1 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined tests.cpp NeuralNetwork.cpp -pthread -o tests-sanitize
	./tests-sanitize

clean:
	rm -f neural01 tests tests-sanitize
