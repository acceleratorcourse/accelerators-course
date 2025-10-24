#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include "group_by.h"
#include <functional>

template <typename KeysIterator,
          typename ValsIterator,
          typename OutputKeysIterator,
          typename OutputValsIterator>
std::pair<OutputKeysIterator, OutputValsIterator> group_by_cpu(KeysIterator a_keys_begin,
                                                               KeysIterator a_keys_end,
                                                               KeysIterator b_keys_begin,
                                                               KeysIterator b_keys_end,
                                                               ValsIterator a_vals_begin,
                                                               ValsIterator b_vals_begin,
                                                               OutputKeysIterator result_keys_begin,
                                                               OutputValsIterator result_vals_begin)
{
    using T = decltype(*a_vals_begin);
    std::multimap<int, T> map;

    for(auto [key_it, val_it] = std::tuple{a_keys_begin, a_vals_begin}; key_it != a_keys_end;
        key_it++, val_it++)
    {
        map.insert(std::pair<int, T>(*key_it, *val_it));
    }

    for(auto [key_it, val_it] = std::tuple{b_keys_begin, b_vals_begin}; key_it != b_keys_end;
        key_it++, val_it++)
    {
        map.insert(std::pair<int, T>(*key_it, *val_it));
    }

    auto key_it = result_keys_begin;
    auto val_it = result_vals_begin;
    for(auto [key, value] : map)
    {
        *key_it++ = key;
        *val_it++ = value;
    }

    return {key_it, val_it};
}

struct GroupByTestCase
{
    size_t vec_size;
    size_t threads_per_block;
};

std::vector<GroupByTestCase> GroupByTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
	    {1,    256},
            {2 << 10, 256},
	    {2 << 12, 256},
            {2 << 19, 32},
            {2 << 19, 64},
            {2 << 19, 128},
            {2 << 19, 256},
            {2 << 19, 512},
    };
    // clang-format on
}

template <typename T = float>
struct GroupByTest : public ::testing::TestWithParam<GroupByTestCase>
{
protected:
    void SetUp() override
    {
        group_by_config = GetParam();
        input_size      = group_by_config.vec_size;

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        a_vals.resize(input_size);
        std::generate(a_vals.begin(), a_vals.end(), gen_value);

        b_vals.resize(input_size);
        std::generate(b_vals.begin(), b_vals.end(), gen_value);

        SetKeys(a_keys);
        SetKeys(b_keys);

        threshold = std::numeric_limits<T>::epsilon() * tolerance;
    }

    void RunCPU()
    {
        auto res = group_by_cpu(a_keys.begin(),
                                a_keys.end(),
                                b_keys.begin(),
                                b_keys.end(),
                                a_vals.begin(),
                                b_vals.begin(),
                                std::back_inserter(cpu_result_keys),
                                std::back_inserter(cpu_result_vals));
    }

    void RunThrust()
    {
        thrust::device_vector<int> dev_a_keys = a_keys;
        thrust::device_vector<T> dev_a_vals   = a_vals;
        thrust::device_vector<int> dev_b_keys = b_keys;
        thrust::device_vector<T> dev_b_vals   = b_vals;
        thrust::device_vector<int> result_keys(input_size * 2);
        thrust::device_vector<T> result_vals(input_size * 2);

        auto res = thrust::merge_by_key(dev_a_keys.begin(),
                                        dev_a_keys.end(),
                                        dev_b_keys.begin(),
                                        dev_b_keys.end(),
                                        dev_a_vals.begin(),
                                        dev_b_vals.begin(),
                                        result_keys.begin(),
                                        result_vals.begin());

        thrust::copy(
            result_vals.begin(), result_vals.end(), std::back_inserter(thrust_result_vals));
        thrust::copy(
            result_keys.begin(), result_keys.end(), std::back_inserter(thrust_result_keys));
    }

    void RunGPU()
    {
        thrust::device_vector<int> dev_a_keys = a_keys;
        thrust::device_vector<T> dev_a_vals   = a_vals;
        thrust::device_vector<int> dev_b_keys = b_keys;
        thrust::device_vector<T> dev_b_vals   = b_vals;
        thrust::device_vector<int> result_keys(input_size * 2);
        thrust::device_vector<T> result_vals(input_size * 2);

        group_by_gpu(dev_a_keys.begin(),
                     dev_a_keys.end(),
                     dev_b_keys.begin(),
                     dev_b_keys.end(),
                     dev_a_vals.begin(),
                     dev_b_vals.begin(),
                     result_keys.begin(),
                     result_vals.begin());

        thrust::copy(result_keys.begin(), result_keys.end(), std::back_inserter(gpu_result_keys));
        thrust::copy(result_vals.begin(), result_vals.end(), std::back_inserter(gpu_result_vals));
    }

    void VerifyCUDA()
    {
        auto val_error = miopen::rms_range(cpu_result_vals, gpu_result_vals);

        EXPECT_TRUE(miopen::range_distance(cpu_result_vals) ==
                    miopen::range_distance(gpu_result_vals));
        EXPECT_TRUE(val_error <= threshold)
            << "GPU vals output do not match CPU vals output. Error:" << val_error
            << " threshold: " << threshold << "\n";

        auto key_error = miopen::rms_range(cpu_result_keys, gpu_result_keys);
        EXPECT_TRUE(miopen::range_distance(cpu_result_keys) ==
                    miopen::range_distance(gpu_result_keys));
        EXPECT_TRUE(key_error <= threshold)
            << "GPU keys output do not match CPU keys output. Error:" << key_error
            << " threshold: " << threshold << "\n";
    }

    void VerifyGPU()
    {
        auto val_error = miopen::rms_range(gpu_result_vals, thrust_result_vals);
        EXPECT_TRUE(miopen::range_distance(gpu_result_vals) ==
                    miopen::range_distance(thrust_result_vals));
        EXPECT_TRUE(val_error <= threshold)
            << "GPU val outputs do not match Thrust val output. Error:" << val_error;

        auto key_error = miopen::rms_range(gpu_result_keys, thrust_result_keys);
        EXPECT_TRUE(miopen::range_distance(gpu_result_keys) ==
                    miopen::range_distance(thrust_result_keys));
        EXPECT_TRUE(key_error <= threshold)
            << "GPU keys outputs do not match Thrust keys output. Error:" << key_error;
    }

    void VerifyCPU()
    {
        auto val_error = miopen::rms_range(cpu_result_vals, thrust_result_vals);

        EXPECT_TRUE(miopen::range_distance(cpu_result_vals) ==
                    miopen::range_distance(thrust_result_vals));
        EXPECT_TRUE(val_error <= threshold)
            << "Thrust vals output do not match CPU vals output. Error:" << val_error;

        auto key_error = miopen::rms_range(cpu_result_keys, thrust_result_keys);
        EXPECT_TRUE(miopen::range_distance(cpu_result_keys) ==
                    miopen::range_distance(thrust_result_keys));
        EXPECT_TRUE(key_error <= threshold)
            << "Thrust keys output do not match CPU keys output. Error:" << key_error;
    }

    void SetKeys(std::vector<int>& keys)
    {
        int size        = input_size;
        int segment_num = 0;
        while(size != 0)
        {
            int segment_size = std::max(rand() % size, 1);
            for(int i = 0; i < segment_size; i++)
            {
                keys.push_back(segment_num);
            }
            segment_num++;
            size -= segment_size;
        }
    }

    GroupByTestCase group_by_config;

    std::vector<int> a_keys;
    std::vector<int> b_keys;

    std::vector<T> a_vals;
    std::vector<T> b_vals;

    std::vector<int> cpu_result_keys;
    std::vector<T> cpu_result_vals;

    std::vector<int> gpu_result_keys;
    std::vector<T> gpu_result_vals;

    std::vector<int> thrust_result_keys;
    std::vector<T> thrust_result_vals;

    size_t input_size;
    double tolerance = 80;
    double threshold;
};

namespace group_by {

struct GroupByTestFloat : GroupByTest<float>
{
};

} // namespace group_by

using namespace group_by;

TEST_P(GroupByTestFloat, GroupByTestFw)
{
    RunCPU();

    RunGPU();

    RunThrust();

    VerifyCUDA();

    VerifyGPU();

    VerifyCPU();
};

INSTANTIATE_TEST_SUITE_P(GroupByTestSet, GroupByTestFloat, testing::ValuesIn(GroupByTestConfigs()));
