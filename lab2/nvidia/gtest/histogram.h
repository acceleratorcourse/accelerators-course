#include <thrust/memory.h>
#include <thrust/device_vector.h>

template<typename T, typename Func>
__global__ void histogram_gpu_atomiic(T* input, int* output, size_t len, Func f, int module)
{
   unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

   if(tid >= len) return;

   int index = std::abs(f(input[tid]))%module;

   atomicAdd((output + index),1);
}

template<typename InputIterator, typename OutputIterator, typename Func>
void histogram_gpu(InputIterator first, InputIterator last, OutputIterator out, Func f, int range, int num_threads)
{
    int len = std::distance(first,last);
    int num_blocks = (len + num_threads - 1)/num_threads;
    histogram_gpu_atomiic<<<dim3(num_blocks, 1, 1), dim3(num_threads, 1, 1), 0, 0>>>(thrust::raw_pointer_cast(&*first),
		                                                                     thrust::raw_pointer_cast(&*out),
										     len,
										     f,
										     range);

}
