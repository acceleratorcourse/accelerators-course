
#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <thrust/functional.h>
#include <thrust/system/hip/execution_policy.h>
#include "histogram.h"

template <typename InputIterator, typename OutputIterator, typename Func>
void histogram_cpu(InputIterator first, InputIterator last, OutputIterator out, Func f, int range)
{
    for(auto it = first; it != last; it++)
    {
        out[std::abs(f(*it)) % range]++;
    }
}

struct HistogramTestCase
{
    size_t vec_size;
    size_t threads_per_block;
};

std::vector<HistogramTestCase> HistogramTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
	    {10,256},
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
struct HistogramTest : public ::testing::TestWithParam<HistogramTestCase>
{
protected:
    void SetUp() override
    {
        histogram_config = GetParam();
        input_size       = histogram_config.vec_size;
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input.resize(input_size);
        std::generate(input.begin(), input.end(), gen_value);
        threshold = std::numeric_limits<T>::epsilon() * tolerance;
    }

    void RunCPU()
    {
        cpu_result.resize(input_size);
        histogram_cpu(
            input.begin(),
            input.end(),
            cpu_result.begin(),
            [](T x) -> int { return (int)10 * x; },
            input_size);
    }

    void RunHIP()
    {
        thrust::device_vector<T> dev_vect = input;
        thrust::device_vector<int> dev_result(input_size, 0);
        histogram_gpu(
            dev_vect.begin(),
            dev_vect.end(),
            dev_result.begin(),
            [](T x) -> int { return (int)10 * x; },
            input_size,
            histogram_config.threads_per_block);
        thrust::copy(dev_result.begin(), dev_result.end(), std::back_inserter(gpu_result));
    }

    void VerifyHIP()
    {
        auto error = miopen::rms_range(cpu_result, gpu_result);
        EXPECT_TRUE(miopen::range_distance(cpu_result) == miopen::range_distance(gpu_result));
        EXPECT_TRUE(error <= threshold) << "GPU output do not match CPU output. Error:" << error
                                        << " threshold: " << threshold << "\n";
    }

    HistogramTestCase histogram_config;
    int input_size;
    std::vector<T> input;
    std::vector<int> cpu_result;
    std::vector<int> gpu_result;
    double tolerance = 80;
    double threshold;
};

namespace histogram {

struct HistogramTestFloat : HistogramTest<float>
{
};

} // namespace histogram
using namespace histogram;

TEST_P(HistogramTestFloat, HistogramTestFw)
{
    RunCPU();

    RunHIP();

    // Verify HIP results against CPU reference
    VerifyHIP();
};

INSTANTIATE_TEST_SUITE_P(HistogramTestSet,
                         HistogramTestFloat,
                         testing::ValuesIn(HistogramTestConfigs()));
