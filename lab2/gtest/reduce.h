#include <thrust/memory.h>

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
   if (bid == 0) {
       for (int i = 0; i < num_blocks; i++) res = binary_op(res,result[i]);
       result[0] = res;
   }
}

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

template<typename T, typename BinaryFunction>
T reduce_gpu_one_block(thrust::device_vector<T>&input, T init, BinaryFunction binary_op, int num_threads) {
   T* out;
   T result;

   int num_blocks = (input.size() + num_threads - 1)/num_threads;
   HIP_CHECK(hipMalloc(&out, sizeof(T) * num_blocks));
   
   int* locks;
   HIP_CHECK(hipMalloc(&locks, sizeof(int) * num_blocks));
   HIP_CHECK(hipMemset(locks, 0, sizeof(int) * num_blocks)); 
   reduce_gpu_one_block<<<dim3(num_blocks, 1, 1), dim3(num_threads, 1, 1), 0, 0>>>(thrust::raw_pointer_cast(input.data()), locks, input.size(), init, binary_op, out);
   HIP_CHECK(hipMemcpy(&result, out, sizeof(T), hipMemcpyDeviceToHost));
   HIP_CHECK(hipFree(out));

   return result;
}
