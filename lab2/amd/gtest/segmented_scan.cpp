#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include "segmented_scan.h"
#include <functional>

template <typename T>
void segmented_scan_cpu(const std::vector<int>& keys,
                        const std::vector<T>& vals,
                        std::vector<T>& out,
                        T init)
{
    if(keys.size() == 0)
        return;

    out.resize(keys.size());

    out[0] = init;

    for(int i = 1; i < keys.size(); i++)
    {
        if(keys[i] == keys[i - 1])
        {
            out[i] = out[i - 1] + vals[i - 1];
        }
        else
        {
            out[i] = init;
        }
    }
}

struct SegmentedPrefixSumTestCase
{
    size_t vec_size;
    size_t threads_per_block;
};

std::vector<SegmentedPrefixSumTestCase> SegmentedPrefixSumTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
	    {1,    256},
	    {256,  256},
            {256,  32},
            {512,  256},
            {512,  32},
            {1024, 256},
            {1024, 32},
            {2048, 64},
    };
    // clang-format on
}

template <typename T = float>
struct SegmentedPrefixSumTest : public ::testing::TestWithParam<SegmentedPrefixSumTestCase>
{
protected:
    void SetUp() override
    {
        segment_scan_config = GetParam();
        input_size          = segment_scan_config.vec_size;

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        vals.resize(input_size);
        std::generate(vals.begin(), vals.end(), gen_value);

        SetKeys();

        threshold = std::numeric_limits<T>::epsilon() * tolerance;
    }

    void RunCPU() { segmented_scan_cpu(keys, vals, cpu_result, (T)0.0); }

    void RunThrust()
    {
        thrust::device_vector<T> dev_keys = keys;
        thrust::device_vector<T> dev_vals = vals;
        thrust::exclusive_scan_by_key(
            dev_keys.begin(), dev_keys.end(), dev_vals.begin(), dev_vals.begin(), (T)0.0);
        thrust::copy(dev_vals.begin(), dev_vals.end(), std::back_inserter(thrust_result));
    }

    void RunHIP()
    {
        thrust::device_vector<T> dev_keys = keys;
        thrust::device_vector<T> dev_vals = vals;
        segmented_scan_gpu(dev_keys.begin(),
                           dev_keys.end(),
                           dev_vals.begin(),
                           dev_vals.begin(),
                           (T)0.0,
                           input_size);
        thrust::copy(dev_vals.begin(), dev_vals.end(), std::back_inserter(gpu_result));
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

    void SetKeys()
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

    void SetKeysTest()
    {
        int segment_num = 0;
        for(int i = 0; i < input_size; i++)
        {
            keys.push_back(segment_num++);
        }
    }

    SegmentedPrefixSumTestCase segment_scan_config;
    std::vector<int> keys;
    std::vector<T> vals;
    std::vector<T> cpu_result;
    std::vector<T> gpu_result;
    std::vector<T> thrust_result;
    size_t input_size;
    double tolerance = 80;
    double threshold;
};

namespace segment_scan {

struct SegmentedPrefixSumTestFloat : SegmentedPrefixSumTest<float>
{
};

} // namespace segment_scan

using namespace segment_scan;

TEST_P(SegmentedPrefixSumTestFloat, SegmentedPrefixSumTestFw)
{
    RunCPU();

    RunHIP();

    RunThrust();

    VerifyHIP();

    VerifyGPU();

    VerifyCPU();
};

INSTANTIATE_TEST_SUITE_P(SegmentedPrefixSumTestSet,
                         SegmentedPrefixSumTestFloat,
                         testing::ValuesIn(SegmentedPrefixSumTestConfigs()));
