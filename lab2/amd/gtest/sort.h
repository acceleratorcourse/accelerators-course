#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/sort.h>

template<typename KeysIterator, typename ValsIterator>
void sort_by_key_gpu(KeysIterator first, KeysIterator last, ValsIterator vals) {
    return thrust::sort_by_key(first, last, vals);
}

