#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

template<typename KeysIterator, typename ValsIterator, typename OutputIterator, typename T>
OutputIterator segmented_scan_gpu(KeysIterator first, KeysIterator last, ValsIterator vals, OutputIterator result, T init, int num_blocks) {
    return thrust::exclusive_scan_by_key(first, last, vals, result, init);
}

