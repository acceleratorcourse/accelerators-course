#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const int page_size = 4096; // Assume the system virtual page size is 4KB
    const int tlb_size = 64; // Assume TLB size is 64 entries
    std::vector<int> data(tlb_size * page_size / sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        for (int j = 0; j < data.size(); j += page_size / sizeof(int)) { // Try different step
            data[j] = 0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}

// g++ -o tlb_cache tlb_cache.cpp
// perf stat -e dTLB-loads,dTLB-load-misses ./tlb_cache

// Observe the changes in dTLB-load-misses and find the number of pages that significantly increases the miss rate, which is the TLB size. By varying the step size, the correlation of the TLB can be estimated.