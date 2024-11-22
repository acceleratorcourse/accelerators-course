#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const int l2_cache_size = 256 * 1024; // Assume L2 cache size is 256KB
    const int llc_cache_size = 8 * 1024 * 1024; // Assume LLC cache size is 8MB
    const int cacheline_size = 64;
    std::vector<int> data(llc_cache_size / cacheline_size * 2);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        for (int j = 0; j < data.size(); j += 16) { // Try different step
            data[j] = 0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}

// g++ -o l2_llc_cache l2_llc_cache.cpp
// perf stat -e L2-loads,L2-load-misses,LLC-loads,LLC-load-misses ./l2_llc_cache

// Observe the changes in L2-load-misses and LLC-load-misses, and find the cache size that significantly increases the miss rate, which is the L2 and LLC cache size. By changing the step size, the cache associativity can be estimated.