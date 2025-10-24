#include <thrust/memory.h>
#include <thrust/device_vector.h>

template<typename T, typename BinaryFunction>
__global__ void prefix_sum(T* input, size_t len, T init, BinaryFunction binary_op, T* result)
{

}

template<typename T, typename Iterator, typename BinaryFunction>
void exclusive_gpu(Iterator begin, Iterator end, Iterator output, T init, BinaryFunction binary_op, int num_threads) {
    thrust::exclusive_scan(begin, end, output, init, binary_op);
}

