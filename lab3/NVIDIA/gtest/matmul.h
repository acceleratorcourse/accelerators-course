#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t err_ = (err);                                              \
        if(err_ != cudaSuccess)                                                \
        {                                                                      \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
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

cublasStatus_t RunGPUImplNaive(thrust::device_vector<float>& A,
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

cublasStatus_t RunGPUImplLDS(thrust::device_vector<float>& A,
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
