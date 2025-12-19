#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

#define WIDTH 96

#include <stdio.h>

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

using namespace std;

unsigned long long dtime_usec(unsigned long long start)
{
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

__global__ void matrixTranspose_static_shared(float* out, float* in, const int width)
{
    __shared__ float sharedMem[WIDTH * WIDTH];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

__global__ void matrixTranspose_dynamic_shared(float* out, float* in, const int width)
{
    extern __shared__ float sharedMem[];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

void MultipleStream(
    float** data, float* randArray, float** gpuTransposeMatrix, float** TransposeMatrix, int width);

void SingleStream(
    float** data, float* randArray, float** gpuTransposeMatrix, float** TransposeMatrix, int width);
int main()
{
    cudaSetDevice(0);

    float *data[2], *TransposeMatrix[2], *gpuTransposeMatrix[2], *randArray;

    int width   = WIDTH;
    size_t size = width * width;

    randArray = (float*)malloc(size * sizeof(float));

    TransposeMatrix[0] = (float*)malloc(size * sizeof(float));
    TransposeMatrix[1] = (float*)malloc(size * sizeof(float));

    cudaMalloc((void**)&gpuTransposeMatrix[0], size * sizeof(float));
    cudaMalloc((void**)&gpuTransposeMatrix[1], size * sizeof(float));

    for(int i = 0; i < size; i++)
    {
        randArray[i] = (float)i * 1.0f;
    }

    unsigned long long multistream_time = dtime_usec(0);
    MultipleStream(data, randArray, gpuTransposeMatrix, TransposeMatrix, width);
    cudaDeviceSynchronize();
    multistream_time = dtime_usec(multistream_time);
    printf("Multistream elapsed time = %fs\n", multistream_time / (float)USECPSEC);

    unsigned long long singlestream_time = dtime_usec(0);
    SingleStream(data, randArray, gpuTransposeMatrix, TransposeMatrix, width);
    cudaDeviceSynchronize();
    singlestream_time = dtime_usec(singlestream_time);
    printf("Singlestream elapsed time = %fs\n", singlestream_time / (float)USECPSEC);

    // verify the results
    int errors = 0;
    double eps = 1.0E-6;
    for(int i = 0; i < size; i++)
    {
        if(std::abs(TransposeMatrix[0][i] - TransposeMatrix[1][i]) > eps)
        {
            printf("%d stream0: %f stream1  %f\n", i, TransposeMatrix[0][i], TransposeMatrix[1][i]);
            errors++;
        }
    }

    if(errors != 0)
    {
        printf("FAILED: %d errors\n", errors);
    }
    else
    {
        printf("stream PASSED!\n");
    }

    free(randArray);
    for(int i = 0; i < 2; i++)
    {
        cudaFree(data[i]);
        cudaFree(gpuTransposeMatrix[i]);
        free(TransposeMatrix[i]);
    }

    cudaDeviceReset();
    return 0;
}

void MultipleStream(
    float** data, float* randArray, float** gpuTransposeMatrix, float** TransposeMatrix, int width)
{
    size_t num_streams = 2;
    size_t size        = width * width;

    cudaStream_t streams[num_streams];

    for(int i = 0; i < num_streams; ++i)
        cudaStreamCreate(&streams[i]);

    for(int i = 0; i < num_streams; i++)
    {
        cudaMalloc((void**)&data[i], size * sizeof(float));
        cudaMemcpyAsync(
            data[i], randArray, size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    matrixTranspose_static_shared<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                                    0,
                                    streams[0]>>>(gpuTransposeMatrix[0], data[0], width);

    matrixTranspose_dynamic_shared<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                                     sizeof(float) * WIDTH * WIDTH,
                                     streams[1]>>>(gpuTransposeMatrix[1], data[1], width);

    for(int i = 0; i < num_streams; i++)
        cudaMemcpyAsync(TransposeMatrix[i],
                        gpuTransposeMatrix[i],
                        size * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[i]);
}

void SingleStream(
    float** data, float* randArray, float** gpuTransposeMatrix, float** TransposeMatrix, int width)
{
    size_t array_size = 2;
    size_t size       = width * width;

    cudaStream_t stream;

    cudaStreamCreate(&stream);

    for(int i = 0; i < array_size; i++)
    {
        cudaMalloc((void**)&data[i], size * sizeof(float));
        cudaMemcpyAsync(data[i], randArray, size * sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    matrixTranspose_static_shared<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                                    0,
                                    stream>>>(gpuTransposeMatrix[0], data[0], width);

    matrixTranspose_dynamic_shared<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                                     sizeof(float) * WIDTH * WIDTH,
                                     stream>>>(gpuTransposeMatrix[1], data[1], width);

    for(int i = 0; i < array_size; i++)
        cudaMemcpyAsync(TransposeMatrix[i],
                        gpuTransposeMatrix[i],
                        size * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream);
}
