#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const int cache_size = 32 * 1024; // Assume L1 cache size is 32KB
    const int cacheline_size = 64;
    std::vector<int> data(cache_size / cacheline_size * 2);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        for (int j = 0; j < data.size(); j += 16) { // try different step
            data[j] = 0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
// g++ -o l1_cache l1_cache.cpp
// perf stat -e cache-references,cache-misses ./l1_cache

// Observe the changes in L1-dcache-load-misses and find the cache size that significantly increases the miss rate, which is the L1 cache size. By changing the step size, the cache associativity can be estimated.