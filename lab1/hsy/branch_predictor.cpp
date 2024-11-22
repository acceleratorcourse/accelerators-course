#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const int array_size = 1000000;
    std::vector<int> data(array_size);

    for (int i = 0; i < array_size; ++i) {
        data[i] = i % 2 == 0 ? 0 : 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        if (data[i % array_size] == 0) {
            sum += data[i % array_size];
        } else {
            sum -= data[i % array_size];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
// g++ -o branch_predictor branch_predictor.cpp
// perf stat -e branch-misses ./branch_predictor

// By observing the changes in branch-misses, you can see the effect of the branch predictor. If the branch predictor works well, branch-misses should be relatively rare.