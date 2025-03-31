#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <rocblas/rocblas.h>

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr, "rocBLAS error: ");           \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

rocblas_status RunGPUImplNaive(thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, rocblas_int m, rocblas_int n, rocblas_int k, float alpha, float beta) {        
        rocblas_handle handle;
        rocblas_status rstatus = rocblas_create_handle(&handle);
        CHECK_ROCBLAS_STATUS(rstatus); 

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        float host_alpha = alpha;
        float host_beta = beta;

        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        rocblas_status status = rocblas_sgemm(
            handle, transA, transB, m, n, k, &host_alpha, thrust::raw_pointer_cast(A.data()), m, thrust::raw_pointer_cast(B.data()), k, &host_beta, thrust::raw_pointer_cast(C.data()), m);
        return status;

}

rocblas_status RunGPUImplLDS(thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, rocblas_int m, rocblas_int n, rocblas_int k, float alpha, float beta) {
        rocblas_handle handle;
        rocblas_status rstatus = rocblas_create_handle(&handle);
        CHECK_ROCBLAS_STATUS(rstatus);

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        float host_alpha = alpha;
        float host_beta = beta;

        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        rocblas_status status = rocblas_sgemm(
            handle, transA, transB, m, n, k, &host_alpha, thrust::raw_pointer_cast(A.data()), m, thrust::raw_pointer_cast(B.data()), k, &host_beta, thrust::raw_pointer_cast(C.data()), m);
        return status;

}

rocblas_status RunGPUImplRegisters(thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, rocblas_int m, rocblas_int n, rocblas_int k, float alpha, float beta) {
        rocblas_handle handle;
        rocblas_status rstatus = rocblas_create_handle(&handle);
        CHECK_ROCBLAS_STATUS(rstatus);

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        float host_alpha = alpha;
        float host_beta = beta;

        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        rocblas_status status = rocblas_sgemm(
            handle, transA, transB, m, n, k, &host_alpha, thrust::raw_pointer_cast(A.data()), m, thrust::raw_pointer_cast(B.data()), k, &host_beta, thrust::raw_pointer_cast(C.data()), m);
        return status;

}

rocblas_status RunGPUImplTensorCore(thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, rocblas_int m, rocblas_int n, rocblas_int k, float alpha, float beta) {
        rocblas_handle handle;
        rocblas_status rstatus = rocblas_create_handle(&handle);
        CHECK_ROCBLAS_STATUS(rstatus);

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        float host_alpha = alpha;
        float host_beta = beta;

        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        rocblas_status status = rocblas_sgemm(
            handle, transA, transB, m, n, k, &host_alpha, thrust::raw_pointer_cast(A.data()), m, thrust::raw_pointer_cast(B.data()), k, &host_beta, thrust::raw_pointer_cast(C.data()), m);
        return status;

}

