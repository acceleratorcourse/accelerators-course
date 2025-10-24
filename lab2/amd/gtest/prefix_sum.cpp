#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/device_vector.h>
#include "prefix_sum.h"
#include <functional>

struct PrefixSumTestCase
{
    size_t vec_size;
    size_t threads_per_block;
};

std::vector<PrefixSumTestCase> PrefixSumTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
	    {256, 256},
            {256, 32},
            {512, 256},
            {512, 32},
            {1024, 256},
            {1024, 32},
            {2048, 64},
    };
    // clang-format on
}

template <typename T = float>
struct PrefixSumTest : public ::testing::TestWithParam<PrefixSumTestCase>
{
protected:
    void SetUp() override
    {
        prefix_sum_config = GetParam();
        auto gen_value    = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input.resize(prefix_sum_config.vec_size);
        std::generate(input.begin(), input.end(), gen_value);
        threshold = std::numeric_limits<T>::epsilon() * tolerance;
    }

    void RunCPU()
    {
        cpu_result.reserve(input.size());
        std::exclusive_scan(
            input.begin(), input.end(), std::back_inserter(cpu_result), (T)0.0, std::plus<T>());
    }

    void RunThrust()
    {
        thrust::device_vector<T> dev_vect = input;
        thrust::exclusive_scan(
            dev_vect.begin(), dev_vect.end(), dev_vect.begin(), (T)0.0, thrust::plus<T>());
        thrust::copy(dev_vect.begin(), dev_vect.end(), std::back_inserter(thrust_result));
    }

    void RunHIP()
    {
        thrust::device_vector<T> dev_vect = input;
        exclusive_gpu(dev_vect.begin(),
                      dev_vect.end(),
                      dev_vect.begin(),
                      (T)0.0,
                      thrust::plus<T>(),
                      prefix_sum_config.threads_per_block);
        thrust::copy(dev_vect.begin(), dev_vect.end(), std::back_inserter(gpu_result));
    }

    void VerifyHIP()
    {
        auto error = miopen::rms_range(cpu_result, gpu_result);
        EXPECT_TRUE(miopen::range_distance(cpu_result) == miopen::range_distance(gpu_result));
        EXPECT_TRUE(error <= threshold) << "GPU output do not match CPU output. Error:" << error
                                        << " threshold: " << threshold << "\n";
    }

    void VerifyGPU()
    {
        auto error = miopen::rms_range(gpu_result, thrust_result);
        EXPECT_TRUE(miopen::range_distance(gpu_result) == miopen::range_distance(thrust_result));
        EXPECT_TRUE(error <= threshold)
            << "GPU outputs do not match Thrust output. Error:" << error;
    }

    void VerifyCPU()
    {
        auto error = miopen::rms_range(cpu_result, thrust_result);
        EXPECT_TRUE(miopen::range_distance(cpu_result) == miopen::range_distance(thrust_result));
        EXPECT_TRUE(error <= threshold) << "Thrust output do not match CPU output. Error:" << error;
    }

    PrefixSumTestCase prefix_sum_config;
    std::vector<T> input;
    std::vector<T> cpu_result;
    std::vector<T> gpu_result;
    std::vector<T> thrust_result;
    double tolerance = 80;
    double threshold;
};

namespace prefix_sum {
struct PrefixSumTestFloat : PrefixSumTest<float>
{
};
} // namespace prefix_sum

using namespace prefix_sum;

TEST_P(PrefixSumTestFloat, PrefixSumTestFw)
{
    RunCPU();

    RunHIP();

    // Verify HIP results against CPU reference
    VerifyHIP();

    RunThrust();

    // Verify Thrust and HIP results against each other
    VerifyGPU();

    VerifyCPU();
};

INSTANTIATE_TEST_SUITE_P(PrefixSumTestSet,
                         PrefixSumTestFloat,
                         testing::ValuesIn(PrefixSumTestConfigs()));
