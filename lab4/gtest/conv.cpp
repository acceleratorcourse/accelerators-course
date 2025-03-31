#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include "conv.h"
#include "tensor.hpp"
#include "data_type.h"
#include <type_traits>
#include <miopen/miopen.h>

#define MIOPEN_CHECK_RET(val) ASSERT_EQ(val, miopenStatusSuccess)

struct ConvTestCase
{
    int n, c, h, w, k, r, s;
    int stride0, stride1;
    int pad0, pad1;
    int dil0, dil1;
    int threads_per_block;
};

template <typename T>
void conv_cpu(TensorDescriptor<T>& in_desc,
              TensorDescriptor<T>& w_desc,
              TensorDescriptor<T>& out_desc,
              const T* input,
              const T* kernel,
              T* output)
{
}

std::vector<ConvTestCase> ConvTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return { 
        {      2,   64,   28,   28,  128,    3,    3,    1,    1,    0,    0,    1,    1,    256},
        {     16,  128,   56,   56,  256,    3,    3,    1,    1,    0,    0,    1,    1,    256},
        {     16,  128,   64,   64,  256,    3,    3,    1,    1,    0,    0,    1,    1,    256},
        {     16,  128,   80,   64,  256,    3,    3,    1,    1,    0,    0,    1,    1,    256},
        {     32,  128,   80,   80,  256,    3,    3,    1,    1,    0,    0,    1,    1,    256},
        {     32,  256,   32,   32,  256,    3,    3,    1,    1,    0,    0,    1,    1,    256},
    };
    // clang-format on
}

template <typename T>
struct ConvTest : public ::testing::TestWithParam<ConvTestCase>
{
protected:
    void SetUp() override
    {
        conv_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        ifm_desc =
            TensorDescriptor<T>({conv_config.n, conv_config.c, conv_config.h, conv_config.w});
        input.resize(ifm_desc.GetElementSize());
        std::generate(input.begin(), input.end(), gen_value);

        ofm_desc = TensorDescriptor<T>(
            {conv_config.k, conv_config.c, CalcOutputHeight(), CalcOutputWidth()});
        output.resize(ofm_desc.GetElementSize());

        kernel_desc =
            TensorDescriptor<T>({conv_config.k, conv_config.c, conv_config.r, conv_config.s});
        kernel.resize(kernel_desc.GetElementSize());
        std::generate(kernel.begin(), kernel.end(), gen_value);

        threshold = std::numeric_limits<T>::epsilon() * tolerance;
    }

    void RunCPU()
    {
        cpu_result.resize(ofm_desc.GetElementSize());
        conv_cpu(ifm_desc, kernel_desc, ofm_desc, input.data(), kernel.data(), cpu_result.data());
    }

    void RunGPUDirect() {}

    void RunGPUWinograd() {}

    void RunGPUImplicitGemm() {}

    void RunGPUFFT() {}
/*
    void RunCuDNN()
    {
        namespace fe = cudnn_frontend;

        auto build_new_graph = [=](cudnnHandle_t handle) {
            auto graph     = std::make_shared<fe::graph::Graph>();
            auto data_type = to_cudnn_data_type<T>::get();

            graph->set_io_data_type(data_type).set_compute_data_type(data_type);

            auto X = graph->tensor(
                fe::graph::Tensor_attributes()
                    .set_name("image")
                    .set_dim({conv_config.n, conv_config.c, conv_config.h, conv_config.w})
                    .set_stride({conv_config.c * conv_config.h * conv_config.w,
                                 1,
                                 conv_config.c * conv_config.w,
                                 conv_config.c}));

            auto W = graph->tensor(
                fe::graph::Tensor_attributes()
                    .set_name("kernel")
                    .set_dim({conv_config.k, conv_config.c, conv_config.r, conv_config.s})
                    .set_stride({conv_config.c * conv_config.r * conv_config.s,
                                 1,
                                 conv_config.c * conv_config.s,
                                 conv_config.c}));

            auto conv_options =
                fe::graph::Conv_fprop_attributes().set_padding(pad).set_stride(stride).set_dilation(
                    dilation);

            auto Y = graph->conv_fprop(X, W, conv_options);

            Y->set_output(true);

            REQUIRE(graph->validate().is_good());

            REQUIRE(graph->build_operation_graph(handle).is_good());

            REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

            REQUIRE(graph->check_support(handle).is_good());

            REQUIRE(graph->build_plans(handle).is_good());

            return std::make_tuple(graph, X, W, Y);
        };
        auto handle_ptr = create_cudnn_handle();
        auto handle     = *handle_ptr;

        auto [graph, X, W, Y] = build_new_graph(handle);

        Surface<T> x_tensor(conv_config.n * conv_config.c * conv_config.h * conv_config.w, false);
        Surface<T> w_tensor(conv_config.k * conv_config.c * conv_config.r * conv_config.s, false);
        Surface<T> y_tensor(conv_config.n * conv_config.k * conv_config.h * conv_config.w,
                            false); // Should be p, q.

        std::unordered_map<int64_t, void*> variant_pack = {{X->get_uid(), x_tensor.devPtr},
                                                           {W->get_uid(), w_tensor.devPtr},
                                                           {Y->get_uid(), y_tensor.devPtr}};

        int64_t workspace_size;
        REQUIRE(graph->get_workspace_size(workspace_size).is_good());
        Surface<int8_t> workspace(workspace_size, false);

        std::cout << *graph << std::endl;

        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    }
*/
    void RunMIOpen()
    {
        thrust::device_vector<T> d_input  = input;
        thrust::device_vector<T> d_kernel = kernel;
        thrust::device_vector<T> d_output = output;

        miopenTensorDescriptor_t mi_input_descr  = nullptr;
        miopenTensorDescriptor_t mi_kernel_descr = nullptr;
        miopenTensorDescriptor_t mi_output_descr = nullptr;
        miopenHandle_t handle                    = nullptr;
        miopenDataType_t data_type               = to_miopen_data_type<T>::get();

        MIOPEN_CHECK_RET(miopenCreate(&handle));

        // Tensor descriptors
        MIOPEN_CHECK_RET(miopenCreateTensorDescriptor(&mi_input_descr));
        MIOPEN_CHECK_RET(miopenSetTensorDescriptor(mi_input_descr,
                                                   data_type,
                                                   ifm_desc.lens.size(),
                                                   ifm_desc.lens.data(),
                                                   ifm_desc.strides.data()));

        MIOPEN_CHECK_RET(miopenCreateTensorDescriptor(&mi_kernel_descr));
        MIOPEN_CHECK_RET(miopenSetTensorDescriptor(mi_kernel_descr,
                                                   data_type,
                                                   kernel_desc.lens.size(),
                                                   kernel_desc.lens.data(),
                                                   kernel_desc.strides.data()));

        MIOPEN_CHECK_RET(miopenCreateTensorDescriptor(&mi_output_descr));
        MIOPEN_CHECK_RET(miopenSetTensorDescriptor(mi_output_descr,
                                                   data_type,
                                                   ofm_desc.lens.size(),
                                                   ofm_desc.lens.data(),
                                                   ofm_desc.strides.data()));

        miopenConvolutionDescriptor_t mi_conv_descr = nullptr;

        // Convolution descriptor
        MIOPEN_CHECK_RET(miopenCreateConvolutionDescriptor(&mi_conv_descr));
        MIOPEN_CHECK_RET(miopenInitConvolutionNdDescriptor(mi_conv_descr,
                                                           pad.size(),
                                                           pad.data(),
                                                           stride.data(),
                                                           dilation.data(),
                                                           miopenConvolution));
        MIOPEN_CHECK_RET(miopenSetConvolutionGroupCount(mi_conv_descr, 1));

        // Workspace
        size_t sz = 0;

        MIOPEN_CHECK_RET(miopenConvolutionForwardGetWorkSpaceSize(
            handle, mi_kernel_descr, mi_input_descr, mi_conv_descr, mi_output_descr, &sz));

        thrust::device_vector<T> wspace = std::vector<T>(sz);

        miopenConvAlgoPerf_t perf_results[10];
        int perf_results_count;

        ASSERT_EQ(
            miopenFindConvolutionForwardAlgorithm(handle,
                                                  mi_input_descr,
                                                  thrust::raw_pointer_cast(d_input.data()),
                                                  mi_kernel_descr,
                                                  thrust::raw_pointer_cast(d_kernel.data()),
                                                  mi_conv_descr,
                                                  mi_output_descr,
                                                  thrust::raw_pointer_cast(d_output.data()),
                                                  sizeof(perf_results) / sizeof(perf_results[0]),
                                                  &perf_results_count,
                                                  perf_results,
                                                  thrust::raw_pointer_cast(wspace.data()),
                                                  wspace.size(),
                                                  true),
            miopenStatusSuccess);
        ASSERT_GT(perf_results_count, 0);

        const float alpha = 1.f;
        const float beta  = 0.f;

        ASSERT_EQ(miopenConvolutionForward(handle,
                                           &alpha,
                                           mi_input_descr,
                                           thrust::raw_pointer_cast(d_input.data()),
                                           mi_kernel_descr,
                                           thrust::raw_pointer_cast(d_kernel.data()),
                                           mi_conv_descr,
                                           perf_results[0].fwd_algo,
                                           &beta,
                                           mi_output_descr,
                                           thrust::raw_pointer_cast(d_output.data()),
                                           thrust::raw_pointer_cast(wspace.data()),
                                           wspace.size() * sizeof(T)),
                  miopenStatusSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        miopen_result.resize(d_output.size());
        thrust::copy(d_output.begin(), d_output.end(), miopen_result.begin());

        // Convolution descriptor
        if(mi_conv_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyConvolutionDescriptor(mi_conv_descr));
        }

        // Tensor descriptors
        if(mi_output_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyTensorDescriptor(mi_output_descr));
        }

        if(mi_kernel_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyTensorDescriptor(mi_kernel_descr));
        }

        if(mi_input_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyTensorDescriptor(mi_input_descr));
        }

        // MIOpen handle
        if(handle != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroy(handle));
        }
    }

    void VerifyDirectGpuAmd()
    {
        auto error = miopen::rms_range(naive_gpu_result, miopen_result);
        EXPECT_TRUE(miopen::range_distance(naive_gpu_result) ==
                    miopen::range_distance(miopen_result));
        EXPECT_TRUE(error <= threshold)
            << "Direct GPU outputs do not match MIOpen output. Error:" << error;
    }

    void VerifyDirectGpuNvidia()
    {
        auto error = miopen::rms_range(naive_gpu_result, cudnn_result);
        EXPECT_TRUE(miopen::range_distance(naive_gpu_result) ==
                    miopen::range_distance(cudnn_result));
        EXPECT_TRUE(error <= threshold)
            << "Direct GPU outputs do not match Cudnn output. Error:" << error;
    }

    void VerifyImplicitGemmAmd()
    {
        auto error = miopen::rms_range(igemm_result, miopen_result);
        EXPECT_TRUE(miopen::range_distance(igemm_result) == miopen::range_distance(miopen_result));
        EXPECT_TRUE(error <= threshold)
            << "Implicit GEMM GPU outputs do not match MIOpen output. Error:" << error;
    }

    void VerifyImplicitGemmNvidia()
    {
        auto error = miopen::rms_range(igemm_result, cudnn_result);
        EXPECT_TRUE(miopen::range_distance(igemm_result) == miopen::range_distance(cudnn_result));
        EXPECT_TRUE(error <= threshold)
            << "Implicit GEMM GPU outputs do not match Cudnn output. Error:" << error;
    }

    void VerifyWinogradAmd()
    {
        auto error = miopen::rms_range(winograd_result, miopen_result);
        EXPECT_TRUE(miopen::range_distance(winograd_result) ==
                    miopen::range_distance(miopen_result));
        EXPECT_TRUE(error <= threshold)
            << "Winograd GPU outputs do not match MIOpen output. Error:" << error;
    }

    void VerifyWinogradNvidia()
    {
        auto error = miopen::rms_range(winograd_result, cudnn_result);
        EXPECT_TRUE(miopen::range_distance(winograd_result) ==
                    miopen::range_distance(cudnn_result));
        EXPECT_TRUE(error <= threshold)
            << "Winograd GPU outputs do not match Cudnn output. Error:" << error;
    }

    void VerifyFftAmd()
    {
        auto error = miopen::rms_range(fft_result, miopen_result);
        EXPECT_TRUE(miopen::range_distance(fft_result) == miopen::range_distance(miopen_result));
        EXPECT_TRUE(error <= threshold)
            << "Fft GPU outputs do not match MIOpen output. Error:" << error;
    }

    void VerifyFftNvidia()
    {
        auto error = miopen::rms_range(fft_result, cudnn_result);
        EXPECT_TRUE(miopen::range_distance(fft_result) == miopen::range_distance(cudnn_result));
        EXPECT_TRUE(error <= threshold)
            << "Fft GPU outputs do not match MIOpen output. Error:" << error;
    }

    void VerifyCPUAmd()
    {
        auto error = miopen::rms_range(cpu_result, miopen_result);
        EXPECT_TRUE(miopen::range_distance(cpu_result) == miopen::range_distance(miopen_result));
        EXPECT_TRUE(error <= threshold) << "CPU output do not match MIOpen output. Error:" << error;
    }

    void VerifyCPUNvidia()
    {
        auto error = miopen::rms_range(cpu_result, cudnn_result);
        EXPECT_TRUE(miopen::range_distance(cpu_result) == miopen::range_distance(cudnn_result));
        EXPECT_TRUE(error <= threshold) << "CPU output do not match Cudnn output. Error:" << error;
    }

    int CalcOutputHeight()
    {
        return (conv_config.h + 2 * pad[0] - dilation[0] * (conv_config.r - 1) - 1) / stride[0] + 1;
    }

    int CalcOutputWidth()
    {
        return (conv_config.w + 2 * pad[1] - dilation[1] * (conv_config.s - 1) - 1) / stride[1] + 1;
    }

    ConvTestCase conv_config;
    std::vector<T> input;
    std::vector<T> kernel;
    std::vector<T> output;

    TensorDescriptor<T> ifm_desc;
    TensorDescriptor<T> ofm_desc;
    TensorDescriptor<T> kernel_desc;

    std::vector<T> cpu_result;
    std::vector<T> naive_gpu_result;
    std::vector<T> igemm_result;
    std::vector<T> winograd_result;
    std::vector<T> fft_result;
    std::vector<T> miopen_result;
    std::vector<T> cudnn_result;

    std::vector<int> pad      = {0, 0};
    std::vector<int> stride   = {1, 1};
    std::vector<int> dilation = {1, 1};

    double tolerance = 80;
    double threshold;
};

using ConvTestFloat = ConvTest<float>;
TEST_P(ConvTestFloat, ConvTestFw)
{
    // RunCPU();

    RunGPUDirect();

    RunGPUWinograd();

    RunGPUImplicitGemm();

    RunGPUFFT();

    RunMIOpen();

    // VerifyCPU();
};

INSTANTIATE_TEST_SUITE_P(ConvTestSet, ConvTestFloat, testing::ValuesIn(ConvTestConfigs()));
