#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "reduce.h"

struct ReduceTestCase
{
    size_t vec_size;
    size_t threads_per_block;
};

std::vector<ReduceTestCase> ReduceTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
	    {256, 256},
            {256, 32},
            {512, 256},
            {512, 32},
    };
    // clang-format on
}

template <typename T = float>
struct ReduceTest : public ::testing::TestWithParam<ReduceTestCase>
{
protected:
    void SetUp() override
    {
        reduce_config  = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input.resize(reduce_config.vec_size);
        std::generate(input.begin(), input.end(), gen_value);
        threshold = std::numeric_limits<T>::epsilon() * tolerance;
    }

    void RunCPU()
    {
        cpu_result = std::accumulate(input.begin(), input.end(), (T)0.0, std::plus<T>());
    }

    void RunThrust()
    {
        thrust::device_vector<T> dev_vect = input;
        thrust_result = thrust::reduce(dev_vect.begin(), dev_vect.end(), (T)0.0, thrust::plus<T>());
    }

    void RunCUDA()
    {
        thrust::device_vector<T> dev_vect = input;
        gpu_result =
            reduce_gpu_atomic(dev_vect, (T)0.0, thrust::plus<T>(), reduce_config.threads_per_block);
    }

    void VerifyCUDA()
    {
        auto error = std::abs(cpu_result - gpu_result);
        EXPECT_TRUE(error <= threshold) << "GPU output do not match CPU output. Error:" << error;
    }

    void VerifyGPU()
    {
        auto error = std::abs(gpu_result - thrust_result);
        EXPECT_TRUE(error <= threshold)
            << "GPU outputs do not match Thrust output. Error:" << error;
    }

    void VerifyCPU()
    {
        auto error = std::abs(cpu_result - thrust_result);
        EXPECT_TRUE(error <= threshold) << "Thrust output do not match CPU output. Error:" << error;
    }

    ReduceTestCase reduce_config;
    std::vector<T> input;
    T cpu_result;
    T gpu_result;
    T thrust_result;
    double tolerance = 80;
    double threshold;
};

namespace reduce {

struct ReduceTestFloat : ReduceTest<float>
{
};

} // namespace reduce
using namespace reduce;

TEST_P(ReduceTestFloat, ReduceTestFw)
{
    RunCPU();

    RunCUDA();

    // Verify CUDA results against CPU reference
    VerifyCUDA();

    RunThrust();

    // Verify Thrust and CUDA results against each other
    VerifyGPU();

    VerifyCPU();
};

INSTANTIATE_TEST_SUITE_P(ReduceTestSet, ReduceTestFloat, testing::ValuesIn(ReduceTestConfigs()));
