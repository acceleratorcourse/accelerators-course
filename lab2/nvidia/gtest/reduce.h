#include <thrust/memory.h>
#include <thrust/device_vector.h>

// Works good on AMD gpu's but not rtx5060
//
template<typename T,  typename BinaryFunction>
__global__ void reduce_gpu_one_block(T* input, int* locks, size_t len, T init, BinaryFunction binary_op, T* result)
{
   __shared__ T sdata[256];
   // each thread loads one element from global to shared mem
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   unsigned int bid = blockIdx.x;
   unsigned int num_blocks = gridDim.x;
   
   if (i >= len) return;

   sdata[tid] = input[i];

   __syncthreads();
   // do reduction in shared mem
   for(unsigned int s=1; s < blockDim.x; s *= 2) {
     if (tid % (2*s) == 0) {
       sdata[tid] = binary_op(sdata[tid],sdata[tid + s]);
     }
     __syncthreads();
   }
   
   if (tid != 0) return;

   result[bid] = sdata[0];

   if (bid == num_blocks - 1) {
       locks[bid] = 1;
   }
   
   while(locks[(bid+1)%num_blocks] != 1);
   locks[bid] = 1;

   T res = init;
   if (bid != 0)
       return;
       
   for (int i = 0; i < num_blocks; i++) res = binary_op(res,result[i]);
       result[0] = res;
}

// This version uses atomic instruction to reduce results from streaming 
// multiprocessor's to gddr
//
template<typename T,  typename BinaryFunction>
__global__ void reduce_gpu_atomic(T* input, size_t len, T init, BinaryFunction binary_op, T* result)
{
   __shared__ T sdata[256];
   // each thread loads one element from global to shared mem
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

   if (i >= len) return;

   sdata[tid] = input[i];

   __syncthreads();
   // do reduction in shared mem
   for(unsigned int s=1; s < blockDim.x; s *= 2) {
     if (tid % (2*s) == 0) {
       sdata[tid] = binary_op(sdata[tid],sdata[tid + s]);
     }
     __syncthreads();
   }

   if (tid != 0) return;
  
   atomicAdd(result, sdata[tid]);
}

#define CUDA_CHECK(expression)                  \
{                                              \
    const cudaError_t status = expression;      \
    if(status != cudaSuccess){                  \
        std::cerr << "CUDA error "              \
                  << status << ": "            \
                  << cudaGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

template<typename T, typename BinaryFunction>
T reduce_gpu_one_block(thrust::device_vector<T>&input, T init, BinaryFunction binary_op, int num_threads) {
   T* out;
   T result;

   int num_blocks = (input.size() + num_threads - 1)/num_threads;
   CUDA_CHECK(cudaMalloc(&out, sizeof(T) * num_blocks));
   
   int* locks;
   CUDA_CHECK(cudaMalloc(&locks, sizeof(int) * num_blocks));
   CUDA_CHECK(cudaMemset(locks, 0, sizeof(int) * num_blocks)); 
   reduce_gpu_one_block<<<dim3(num_blocks, 1, 1), dim3(num_threads, 1, 1), 0, 0>>>(thrust::raw_pointer_cast(input.data()), locks, input.size(), init, binary_op, out);
   CUDA_CHECK(cudaMemcpy(&result, out, sizeof(T), cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaFree(out));
   CUDA_CHECK(cudaFree(locks));

   return result;
}

template<typename T, typename BinaryFunction>
T reduce_gpu_atomic(thrust::device_vector<T>&input, T init, BinaryFunction binary_op, int num_threads) {
   T* out;
   T result;

   int num_blocks = (input.size() + num_threads - 1)/num_threads;
   CUDA_CHECK(cudaMalloc(&out, sizeof(T) * num_blocks));
   reduce_gpu_atomic<<<dim3(num_blocks, 1, 1), dim3(num_threads, 1, 1), 0, 0>>>(thrust::raw_pointer_cast(input.data()), input.size(), init, binary_op, out);
   CUDA_CHECK(cudaMemcpy(&result, out, sizeof(T), cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaFree(out));

   return result;
}

