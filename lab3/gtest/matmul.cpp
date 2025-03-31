#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include "matmul.h"
#include <cblas.h>

/*
 *  performs matrix multiply C = alpha*A*B + beta*C
 *  C - matrix with dimensions m x n
 *  A - matrix with dimensions m x k
 *  B - matrix with dimensions n x k
 */

struct GEMMTestCase
{
    size_t m, n, k;
    float alpha, beta;
    size_t threads_per_block;
};

std::vector<GEMMTestCase> GEMMTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
	    {256,256,256, 1.0, 0.0, 256},
            {256,256,256, 1.0, 0.0, 32},
            {512, 512, 512, 1.0, 0.0, 256},
            {512, 512, 512, 1.0, 0.0, 32},
            {1024, 1024, 1024, 1.0, 0.0, 256},
            {1024, 1024, 1024, 1.0, 0.0, 32},
            {2048, 2048, 2048, 1.0, 0.0, 64},
    };
    // clang-format on
}

struct GEMMTest : public ::testing::TestWithParam<GEMMTestCase>
{
protected:
    void SetUp() override
    {
        gemm_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<float>(1e-2, 100); };

        A.resize(gemm_config.m * gemm_config.k);
        std::generate(A.begin(), A.end(), gen_value);

        B.resize(gemm_config.k * gemm_config.n);
        std::generate(B.begin(), B.end(), gen_value);

        C.resize(gemm_config.m * gemm_config.n);
        std::generate(C.begin(), C.end(), gen_value);

        threshold = std::numeric_limits<float>::epsilon() * tolerance;
    }

    void RunBlasCPU()
    {
        cblas_sgemm(CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    gemm_config.m,
                    gemm_config.n,
                    gemm_config.k,
                    gemm_config.alpha,
                    A.data(),
                    gemm_config.k,
                    B.data(),
                    gemm_config.n,
                    gemm_config.beta,
                    C.data(),
                    gemm_config.n);

        std::copy(C.begin(), C.end(), std::back_inserter(blas_cpu_result));
    }

    void RunBlasGPU()
    {
        // using rocblas API
        rocblas_handle handle;
        rocblas_status rstatus = rocblas_create_handle(&handle);
        CHECK_ROCBLAS_STATUS(rstatus);

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        thrust::device_vector<float> dev_A = A;
        thrust::device_vector<float> dev_B = B;
        thrust::device_vector<float> dev_C = C;

        rocblas_int m = gemm_config.m;
        rocblas_int n = gemm_config.n;
        rocblas_int k = gemm_config.k;

        float host_alpha = gemm_config.alpha;
        float host_beta  = gemm_config.beta;

        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        rocblas_status status = rocblas_sgemm(handle,
                                              transA,
                                              transB,
                                              m,
                                              n,
                                              k,
                                              &host_alpha,
                                              thrust::raw_pointer_cast(dev_A.data()),
                                              m,
                                              thrust::raw_pointer_cast(dev_B.data()),
                                              k,
                                              &host_beta,
                                              thrust::raw_pointer_cast(dev_C.data()),
                                              m);
        CHECK_ROCBLAS_STATUS(rstatus);

        thrust::copy(dev_C.begin(), dev_C.end(), std::back_inserter(blas_gpu_result));
    }

    void RunGPUNaive()
    {
        thrust::device_vector<float> dev_A = A;
        thrust::device_vector<float> dev_B = B;
        thrust::device_vector<float> dev_C = C;
        rocblas_status status              = RunGPUImplNaive(dev_A,
                                                dev_B,
                                                dev_C,
                                                gemm_config.m,
                                                gemm_config.n,
                                                gemm_config.k,
                                                gemm_config.alpha,
                                                gemm_config.beta);
        CHECK_ROCBLAS_STATUS(status);
        thrust::copy(dev_C.begin(), dev_C.end(), std::back_inserter(naive_gpu_result));
    }

    void RunGPULDS()
    {
        thrust::device_vector<float> dev_A = A;
        thrust::device_vector<float> dev_B = B;
        thrust::device_vector<float> dev_C = C;
        rocblas_status status              = RunGPUImplLDS(dev_A,
                                              dev_B,
                                              dev_C,
                                              gemm_config.m,
                                              gemm_config.n,
                                              gemm_config.k,
                                              gemm_config.alpha,
                                              gemm_config.beta);
        CHECK_ROCBLAS_STATUS(status);
        thrust::copy(dev_C.begin(), dev_C.end(), std::back_inserter(lds_gpu_result));
    }

    void RunGPURegisters()
    {
        thrust::device_vector<float> dev_A = A;
        thrust::device_vector<float> dev_B = B;
        thrust::device_vector<float> dev_C = C;
        rocblas_status status              = RunGPUImplRegisters(dev_A,
                                                    dev_B,
                                                    dev_C,
                                                    gemm_config.m,
                                                    gemm_config.n,
                                                    gemm_config.k,
                                                    gemm_config.alpha,
                                                    gemm_config.beta);
        CHECK_ROCBLAS_STATUS(status);
        thrust::copy(dev_C.begin(), dev_C.end(), std::back_inserter(registers_gpu_result));
    }

    void RunGPUTensorCore()
    {
        thrust::device_vector<float> dev_A = A;
        thrust::device_vector<float> dev_B = B;
        thrust::device_vector<float> dev_C = C;
        rocblas_status status              = RunGPUImplTensorCore(dev_A,
                                                     dev_B,
                                                     dev_C,
                                                     gemm_config.m,
                                                     gemm_config.n,
                                                     gemm_config.k,
                                                     gemm_config.alpha,
                                                     gemm_config.beta);
        CHECK_ROCBLAS_STATUS(status);
        thrust::copy(dev_C.begin(), dev_C.end(), std::back_inserter(tensor_core_gpu_result));
    }

    void VerifyNaiveGPU()
    {
        auto error = miopen::rms_range(naive_gpu_result, blas_gpu_result);
        EXPECT_TRUE(miopen::range_distance(naive_gpu_result) ==
                    miopen::range_distance(blas_gpu_result));
        EXPECT_TRUE(error <= threshold)
            << "Naive GPU outputs do not match BLAS output. Error:" << error;
    }

    void VerifyTensorCoreGPU()
    {
        auto error = miopen::rms_range(tensor_core_gpu_result, blas_gpu_result);
        EXPECT_TRUE(miopen::range_distance(tensor_core_gpu_result) ==
                    miopen::range_distance(blas_gpu_result));
        EXPECT_TRUE(error <= threshold)
            << "Tensor Core GPU outputs do not match BLAS output. Error:" << error;
    }

    void VerifyLDSGPU()
    {
        auto error = miopen::rms_range(lds_gpu_result, blas_gpu_result);
        EXPECT_TRUE(miopen::range_distance(lds_gpu_result) ==
                    miopen::range_distance(blas_gpu_result));
        EXPECT_TRUE(error <= threshold)
            << "LDS GPU outputs do not match BLAS output. Error:" << error;
    }

    void VerifyRegistersGPU()
    {
        auto error = miopen::rms_range(registers_gpu_result, blas_gpu_result);
        EXPECT_TRUE(miopen::range_distance(registers_gpu_result) ==
                    miopen::range_distance(blas_gpu_result));
        EXPECT_TRUE(error <= threshold)
            << "Registers GPU outputs do not match BLAS output. Error:" << error;
    }

    void VerifyCPU()
    {
        auto error = miopen::rms_range(blas_cpu_result, blas_gpu_result);
        EXPECT_TRUE(miopen::range_distance(blas_gpu_result) ==
                    miopen::range_distance(blas_cpu_result));
        EXPECT_TRUE(error <= threshold) << "BLAS output do not match CPU output. Error:" << error;
    }

    GEMMTestCase gemm_config;
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> C;
    std::vector<float> naive_gpu_result;
    std::vector<float> lds_gpu_result;
    std::vector<float> registers_gpu_result;
    std::vector<float> tensor_core_gpu_result;

    std::vector<float> blas_cpu_result;
    std::vector<float> blas_gpu_result;
    double tolerance = 80;
    double threshold;
};

namespace gemm {
struct GEMMTestFloat : GEMMTest
{
};
} // namespace gemm

using namespace gemm;

TEST_P(GEMMTestFloat, GEMMTestFw)
{
    RunBlasCPU();

    RunBlasGPU();

    VerifyCPU();

    RunGPUNaive();

    RunGPULDS();

    RunGPURegisters();

    RunGPUTensorCore();
};

INSTANTIATE_TEST_SUITE_P(GEMMTestSet, GEMMTestFloat, testing::ValuesIn(GEMMTestConfigs()));
