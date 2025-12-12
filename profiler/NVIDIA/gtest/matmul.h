#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuda.h>
#include <stdio.h>

#define BLOCK_SIZE 16

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

__global__ void NaiveSgemm(
    float* A, float* B, float* C, const int m, const int n, const int k, float alpha, float beta)
{

    float acc = 0;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;

    for(int e = 0; e < k; ++e)
    {
        acc += A[row * m + e] * B[e * n + col];
    }

    C[row * n + col] = beta * C[row * n + col] + alpha * acc;
}

__global__ void
NaiveSgemmNoBeta(float* A, float* B, float* C, const int m, const int n, const int k, float alpha)
{

    float acc = 0;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;

    for(int e = 0; e < k; ++e)
    {
        acc += A[row * m + e] * B[e * n + col];
    }

    C[row * n + col] = alpha * acc;
}

__global__ void SgemmLDS(
    float* A, float* B, float* C, const int m, const int n, const int k, float alpha, float beta)
{

    float acc = 0;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;

    assert(k % blockDim.x == 0);

    for(int i = 0; i < k / BLOCK_SIZE; ++i)
    {
        __shared__ float A_s[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float B_s[BLOCK_SIZE][BLOCK_SIZE];

        A_s[threadIdx.y][threadIdx.x] = A[row * m + col];
        B_s[threadIdx.y][threadIdx.x] = B[row * n + col];

        __syncthreads();

        for(int j = 0; j < k / BLOCK_SIZE; ++j)
        {
            acc += A_s[threadIdx.y][j] * B_s[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * n + col] = beta * C[row * n + col] + alpha * acc;
}

cublasStatus_t RunGPUImplNaive(thrust::device_vector<float>& A,
                               thrust::device_vector<float>& B,
                               thrust::device_vector<float>& C,
                               const int m,
                               const int n,
                               const int k,
                               float alpha,
                               float beta,
                               const int threads_per_block_x,
                               const int threads_per_block_y)
{
    dim3 dimBlock(threads_per_block_x, threads_per_block_y);
    dim3 dimGrid((n + threads_per_block_x - 1) / threads_per_block_x,
                 (m + threads_per_block_y - 1) / threads_per_block_y);

    if(beta != 0.0)
    {
        NaiveSgemm<<<dimGrid, dimBlock, 0, 0>>>(thrust::raw_pointer_cast(B.data()),
                                                thrust::raw_pointer_cast(A.data()),
                                                thrust::raw_pointer_cast(C.data()),
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                beta);
    }
    else
    {
        NaiveSgemmNoBeta<<<dimGrid, dimBlock, 0, 0>>>(thrust::raw_pointer_cast(B.data()),
                                                      thrust::raw_pointer_cast(A.data()),
                                                      thrust::raw_pointer_cast(C.data()),
                                                      m,
                                                      n,
                                                      k,
                                                      alpha);
    }
    CUDA_CHECK(cudaGetLastError());
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t RunGPUImplLDS(thrust::device_vector<float>& A,
                             thrust::device_vector<float>& B,
                             thrust::device_vector<float>& C,
                             const int m,
                             const int n,
                             const int k,
                             float alpha,
                             float beta)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    SgemmLDS<<<dimGrid, dimBlock, 0, 0>>>(thrust::raw_pointer_cast(B.data()),
                                          thrust::raw_pointer_cast(A.data()),
                                          thrust::raw_pointer_cast(C.data()),
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          beta);
    CUDA_CHECK(cudaGetLastError());
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t RunGPUImplRegisters(thrust::device_vector<float>& A,
                                   thrust::device_vector<float>& B,
                                   thrust::device_vector<float>& C,
                                   const int m,
                                   const int n,
                                   const int k,
                                   float alpha,
                                   float beta)
{
    cublasHandle_t handle;
    cublasStatus_t rstatus = cublasCreate(&handle);
    CUBLAS_CHECK(rstatus);

    const cublasOperation_t transA = CUBLAS_OP_N;
    const cublasOperation_t transB = CUBLAS_OP_N;

    float host_alpha = alpha;
    float host_beta  = beta;

    rstatus = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_CHECK(rstatus);

    cublasStatus_t status = cublasSgemm(handle,
                                        transA,
                                        transB,
                                        m,
                                        n,
                                        k,
                                        &host_alpha,
                                        thrust::raw_pointer_cast(A.data()),
                                        m,
                                        thrust::raw_pointer_cast(B.data()),
                                        k,
                                        &host_beta,
                                        thrust::raw_pointer_cast(C.data()),
                                        m);
    return status;
}

cublasStatus_t RunGPUImplTensorCore(thrust::device_vector<float>& A,
                                    thrust::device_vector<float>& B,
                                    thrust::device_vector<float>& C,
                                    const int m,
                                    const int n,
                                    const int k,
                                    float alpha,
                                    float beta)
{
    cublasHandle_t handle;
    cublasStatus_t rstatus = cublasCreate(&handle);
    CUBLAS_CHECK(rstatus);

    const cublasOperation_t transA = CUBLAS_OP_N;
    const cublasOperation_t transB = CUBLAS_OP_N;

    float host_alpha = alpha;
    float host_beta  = beta;

    rstatus = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_CHECK(rstatus);

    cublasStatus_t status = cublasSgemm(handle,
                                        transA,
                                        transB,
                                        m,
                                        n,
                                        k,
                                        &host_alpha,
                                        thrust::raw_pointer_cast(A.data()),
                                        m,
                                        thrust::raw_pointer_cast(B.data()),
                                        k,
                                        &host_beta,
                                        thrust::raw_pointer_cast(C.data()),
                                        m);
    return status;
}
