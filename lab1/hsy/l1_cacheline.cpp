#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const int max_offset = 1024;
    std::vector<int> data(max_offset * 2);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        for (int offset = 0; offset < max_offset; offset += 8) { // Try different step
            data[offset] = 0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
// g++ -o l1_cacheline l1_cacheline.cpp
// perf stat -e cache-references,cache-misses ./l1_cacheline

// Observe the changes in L1-dcache-load-misses and find the minimum step size that significantly increases the miss rate, which is the L1 cache line size.