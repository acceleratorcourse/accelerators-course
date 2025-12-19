#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include "skewing.h"
#include <cblas.h>

#define GRID_SIZE 4

struct StreamTestCase
{
    size_t cell_size;
    float alpha = 1.0;
    float beta  = 1.0;
};

std::vector<StreamTestCase> StreamTestConfigs()
{ // vector_size, threads_per_block
    // clang-format off
    return {
            {1024, 1.0, 0.0},
    };
    // clang-format on
}

struct StreamTest : public ::testing::TestWithParam<StreamTestCase>
{
protected:
    void SetUp() override
    {
        stream_config  = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<float>(1e-2, 100); };

        for(int i = 0; i < GRID_SIZE; i++)
        {
            for(int j = 0; j < GRID_SIZE; j++)
            {
                grid[i][j].resize(stream_config.cell_size * stream_config.cell_size);
                std::generate(grid[i][j].begin(), grid[i][j].end(), gen_value);
                dev_grid[i][j] = grid[i][j];
            }
        }

        threshold = std::numeric_limits<float>::epsilon() * tolerance;
    }

    void RunSequential()
    {
        cublasHandle_t handle;
        cublasStatus_t rstatus = cublasCreate(&handle);
        CUBLAS_CHECK(rstatus);

        const cublasOperation_t transA = CUBLAS_OP_N;
        const cublasOperation_t transB = CUBLAS_OP_N;

        float host_alpha = stream_config.alpha;
        float host_beta  = stream_config.beta;

        rstatus = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        CUBLAS_CHECK(rstatus);

        unsigned long long sequential_time = dtime_usec(0);

        for(int i = 1; i < GRID_SIZE; i++)
        {
            for(int j = 1; j < GRID_SIZE; j++)
            {
                cublasStatus_t status =
                    cublasSgemm(handle,
                                transA,
                                transB,
                                stream_config.cell_size,
                                stream_config.cell_size,
                                stream_config.cell_size,
                                &host_alpha,
                                thrust::raw_pointer_cast(dev_grid[i - 1][j].data()),
                                stream_config.cell_size,
                                thrust::raw_pointer_cast(dev_grid[i][j - 1].data()),
                                stream_config.cell_size,
                                &host_beta,
                                thrust::raw_pointer_cast(dev_grid[i][j].data()),
                                stream_config.cell_size);
                CUBLAS_CHECK(rstatus);
            }
        }

        cudaError_t cuda_status = cudaDeviceSynchronize();
        CUDA_CHECK(cuda_status);

        sequential_time = dtime_usec(sequential_time);
        printf("Sequential  elapsed time = %fs\n", sequential_time / (float)USECPSEC);

        if(GRID_SIZE > 1)
        {
            thrust::copy(dev_grid[GRID_SIZE - 1][GRID_SIZE - 1].begin(),
                         dev_grid[GRID_SIZE - 1][GRID_SIZE - 1].end(),
                         std::back_inserter(sequential_result));
        }
    }

    void RunSkewing()
    {
        cudaStream_t streams[2 * GRID_SIZE - 1];
        cublasHandle_t handles[2 * GRID_SIZE - 1];
        cudaEvent_t ready_events[GRID_SIZE][GRID_SIZE];
        cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;
        cudaError_t cuda_status    = cudaSuccess;

        const cublasOperation_t transA = CUBLAS_OP_N;
        const cublasOperation_t transB = CUBLAS_OP_N;

        float host_alpha = stream_config.alpha;
        float host_beta  = stream_config.beta;

        for(int i = 0; i < 2 * GRID_SIZE - 1; i++)
        {
            cuda_status = cudaStreamCreate(&streams[i]);
            CUDA_CHECK(cuda_status);
            blas_status = cublasCreate(&handles[i]);
            CUBLAS_CHECK(blas_status);
            blas_status = cublasSetPointerMode(handles[i], CUBLAS_POINTER_MODE_HOST);
            CUBLAS_CHECK(blas_status);
        }

        for(int i = 0; i < GRID_SIZE; i++)
        {
            for(int j = 0; j < GRID_SIZE; j++)
            {
                cuda_status = cudaEventCreate(&ready_events[i][j]);
                CUDA_CHECK(cuda_status);
            }
        }

        for(int i = 0; i < GRID_SIZE; i++)
        {
            cuda_status = cudaEventRecord(ready_events[0][i], streams[i]);
            CUDA_CHECK(cuda_status);
            cuda_status = cudaEventRecord(ready_events[i][0], streams[i]);
            CUDA_CHECK(cuda_status);
        }

        unsigned long long skewing_time = dtime_usec(0);

        for(int i = 1; i < GRID_SIZE; i++)
        {
            for(int j = 1; j < GRID_SIZE; j++)
            {
                cublasSetStream(handles[i + j], streams[i + j]);
                CUBLAS_CHECK(blas_status);
                cuda_status = cudaStreamWaitEvent(streams[i + j], ready_events[i - 1][j], 0);
                CUDA_CHECK(cuda_status);
                cuda_status = cudaStreamWaitEvent(streams[i + j], ready_events[i][j - 1], 0);
                CUDA_CHECK(cuda_status);

                cublasStatus_t status =
                    cublasSgemm(handles[i + j],
                                transA,
                                transB,
                                stream_config.cell_size,
                                stream_config.cell_size,
                                stream_config.cell_size,
                                &host_alpha,
                                thrust::raw_pointer_cast(dev_grid[i - 1][j].data()),
                                stream_config.cell_size,
                                thrust::raw_pointer_cast(dev_grid[i][j - 1].data()),
                                stream_config.cell_size,
                                &host_beta,
                                thrust::raw_pointer_cast(dev_grid[i][j].data()),
                                stream_config.cell_size);
                CUDA_CHECK(cuda_status);

                cuda_status = cudaEventRecord(ready_events[i][j], streams[i + j]);
                CUDA_CHECK(cuda_status);
            }
        }

        cuda_status = cudaDeviceSynchronize();
        CUDA_CHECK(cuda_status);

        skewing_time = dtime_usec(skewing_time);
        printf("Skewing_time  elapsed time = %fs\n", skewing_time / (float)USECPSEC);

        for(int i = 0; i < GRID_SIZE; i++)
        {
            for(int j = 0; j < GRID_SIZE; j++)
            {
                cuda_status = cudaEventDestroy(ready_events[i][j]);
                CUDA_CHECK(cuda_status);
            }
        }

        for(int i = 0; i < 2 * GRID_SIZE - 1; i++)
        {
            cuda_status = cudaStreamDestroy(streams[i]);
            CUDA_CHECK(cuda_status);
            blas_status = cublasDestroy(handles[i]);
            CUBLAS_CHECK(blas_status);
        }

        if(GRID_SIZE > 1)
        {
            thrust::copy(dev_grid[GRID_SIZE - 1][GRID_SIZE - 1].begin(),
                         dev_grid[GRID_SIZE - 1][GRID_SIZE - 1].end(),
                         std::back_inserter(skewing_result));
        }
    }

    void RunGraph()
    {
        // using cublas API
        cublasHandle_t handle;
        cudaGraph_t initGraph;
        cudaGraphExec_t execGraph;

        cudaStream_t blas_stream, graph_stream;
        cudaError_t cuda_status = cudaStreamCreate(&graph_stream);
        CUDA_CHECK(cuda_status);

        cuda_status = cudaStreamCreate(&blas_stream);
        CUDA_CHECK(cuda_status);

        cublasStatus_t rstatus = cublasCreate(&handle);
        CUBLAS_CHECK(rstatus);

        rstatus = cublasSetStream(handle, blas_stream);
        CUBLAS_CHECK(rstatus);

        cuda_status = cudaStreamBeginCapture(blas_stream, cudaStreamCaptureModeGlobal);
        CUDA_CHECK(cuda_status);

        const cublasOperation_t transA = CUBLAS_OP_N;
        const cublasOperation_t transB = CUBLAS_OP_N;

        float host_alpha = stream_config.alpha;
        float host_beta  = stream_config.beta;

        rstatus = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        CUBLAS_CHECK(rstatus);

        for(int i = 1; i < GRID_SIZE; i++)
        {
            for(int j = 1; j < GRID_SIZE; j++)
            {
                cublasStatus_t status =
                    cublasSgemm(handle,
                                transA,
                                transB,
                                stream_config.cell_size,
                                stream_config.cell_size,
                                stream_config.cell_size,
                                &host_alpha,
                                thrust::raw_pointer_cast(dev_grid[i - 1][j].data()),
                                stream_config.cell_size,
                                thrust::raw_pointer_cast(dev_grid[i][j - 1].data()),
                                stream_config.cell_size,
                                &host_beta,
                                thrust::raw_pointer_cast(dev_grid[i][j].data()),
                                stream_config.cell_size);
                CUBLAS_CHECK(rstatus);
            }
        }

        cuda_status = cudaStreamEndCapture(blas_stream, &initGraph);
        CUDA_CHECK(cuda_status);

        cuda_status = cudaGraphInstantiate(&execGraph, initGraph, NULL, NULL, 0);
        CUDA_CHECK(cuda_status);

        unsigned long long graph_time = dtime_usec(0);

        cuda_status = cudaGraphLaunch(execGraph, graph_stream);
        CUDA_CHECK(cuda_status);

        cuda_status = cudaStreamSynchronize(graph_stream);
        CUDA_CHECK(cuda_status);

        graph_time = dtime_usec(graph_time);

        printf("graph  elapsed time = %fs\n", graph_time / (float)USECPSEC);

        if(GRID_SIZE > 1)
        {
            thrust::copy(dev_grid[GRID_SIZE - 1][GRID_SIZE - 1].begin(),
                         dev_grid[GRID_SIZE - 1][GRID_SIZE - 1].end(),
                         std::back_inserter(graph_result));
        }
    }

    void VerifySkewing()
    {
        auto error = miopen::rms_range(skewing_result, sequential_result);

        EXPECT_TRUE(miopen::range_distance(skewing_result) ==
                    miopen::range_distance(sequential_result));
        EXPECT_TRUE(error <= threshold)
            << "Stream output do not match Sequential output. Error:" << error;
    }

    void VerifyGraph()
    {
        auto error = miopen::rms_range(graph_result, sequential_result);
        EXPECT_TRUE(miopen::range_distance(graph_result) ==
                    miopen::range_distance(sequential_result));
        EXPECT_TRUE(error <= threshold)
            << "Stream output do not match Sequential output. Error:" << error;
    }

    StreamTestCase stream_config;
    std::vector<float> grid[GRID_SIZE][GRID_SIZE];
    thrust::device_vector<float> dev_grid[GRID_SIZE][GRID_SIZE];
    std::vector<float> sequential_result;
    std::vector<float> skewing_result;
    std::vector<float> graph_result;

    double tolerance = 80;
    double threshold;
};

namespace stream {
struct StreamTestFloat : StreamTest
{
};
} // namespace stream

using namespace stream;

TEST_P(StreamTestFloat, StreamTestFw)
{
    RunSequential();
    RunSkewing();
    RunGraph();
    VerifySkewing();
    VerifyGraph();
};

INSTANTIATE_TEST_SUITE_P(StreamTestSet, StreamTestFloat, testing::ValuesIn(StreamTestConfigs()));
