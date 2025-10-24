#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include "sort.h"
#include <functional>

template <typename KeysIterator, typename ValsIterator>
void sort_by_key_cpu(KeysIterator first, KeysIterator last, ValsIterator vals, ValsIterator out)
{
    for(auto [key_it, val_it] = std::tuple(first, vals); key_it != last; key_it++, val_it++)
    {
        out[*key_it] = *val_it;
    }
}

struct SortTestCase
{
    size_t vec_size;
    size_t threads_per_block;
};

std::vector<SortTestCase> SortTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
	    {1,    256},
            {2 << 19, 32},
            {2 << 19, 64},
            {2 << 19, 128},
            {2 << 19, 256},
            {2 << 19, 512},
    };
    // clang-format on
}

template <typename T = float>
struct SortTest : public ::testing::TestWithParam<SortTestCase>
{
protected:
    void SetUp() override
    {
        sort_config = GetParam();
        input_size  = sort_config.vec_size;

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        vals.resize(input_size);
        std::generate(vals.begin(), vals.end(), gen_value);

        SetKeys();

        threshold = std::numeric_limits<T>::epsilon() * tolerance;
    }

    void RunCPU()
    {
        cpu_result.resize(input_size);
        sort_by_key_cpu(keys.begin(), keys.end(), vals.begin(), cpu_result.begin());
    }

    void RunThrust()
    {
        thrust::device_vector<T> dev_keys = keys;
        thrust::device_vector<T> dev_vals = vals;
        thrust::sort_by_key(dev_keys.begin(), dev_keys.end(), dev_vals.begin());
        thrust::copy(dev_vals.begin(), dev_vals.end(), std::back_inserter(thrust_result));
    }

    void RunCUDA()
    {
        thrust::device_vector<T> dev_keys = keys;
        thrust::device_vector<T> dev_vals = vals;
        sort_by_key_gpu(dev_keys.begin(), dev_keys.end(), dev_vals.begin());
        thrust::copy(dev_vals.begin(), dev_vals.end(), std::back_inserter(gpu_result));
    }

    void VerifyCUDA()
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

    void SetKeys()
    {
        keys.resize(input_size);
        for(int i = 0; i < input_size; i++)
            keys[i] = i;
        std::random_shuffle(keys.begin(), keys.end());
    }

    SortTestCase sort_config;
    std::vector<int> keys;
    std::vector<T> vals;
    std::vector<T> cpu_result;
    std::vector<T> gpu_result;
    std::vector<T> thrust_result;
    size_t input_size;
    double tolerance = 80;
    double threshold;
};

namespace sort {

struct SortTestFloat : SortTest<float>
{
};

} // namespace sort

using namespace sort;

TEST_P(SortTestFloat, SortTestFw)
{
    RunCPU();

    RunCUDA();

    RunThrust();

    VerifyCUDA();

    VerifyGPU();

    VerifyCPU();
};

INSTANTIATE_TEST_SUITE_P(SortTestSet, SortTestFloat, testing::ValuesIn(SortTestConfigs()));
