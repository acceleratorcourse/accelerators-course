#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuda.h>
#include <stdio.h>

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start)
{
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err_ = (err);                                                                  \
        if(err_ != cudaSuccess && err_ != cudaErrorNoKernelImageForDevice)                         \
        {                                                                                          \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while(0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                        \
    do                                                                           \
    {                                                                            \
        cublasStatus_t err_ = (err);                                             \
        if(err_ != CUBLAS_STATUS_SUCCESS)                                        \
        {                                                                        \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                            \
        }                                                                        \
    } while(0)
