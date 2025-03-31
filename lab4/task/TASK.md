# Lab4

Task 1:
Implement conv_cpu function using zendnn or mkl forward convolution functions
Naive cpu implementation is too slow

Task 2:
Implement direct convolution method in RunGPUDirect function
Write result to naive_gpu_result

Task 3:
Implement winograd transform method for convolution in RunGPUWinograd function
Write result to winograd_result

Task 3:
Implement implicit gemm method for convolution in RunGPUImplicitGemm function
Write result to igemm_result

Task 4:
Implement Fast Fourier transform method for convolution in RunGPUFFT function
Write result to fft_result

Task 5 for nvidia platform:
Implement convolution using [cudnn legacy api](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#cudnnconvolutionforward)
Implement convolution using [cudnn backend api](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html#)
Compare it with provided [frontend api realization](https://github.com/NVIDIA/cudnn-frontend)

Task 6:
Run tests with different tensor,kernel shapes, strides
Try to figure out which method is returned via [cudnnFindConvolutionForwardAlgorithim](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#cudnnfindconvolutionforwardalgorithm)
in different cases. Same with [miopenFindConvolutionForwardAlgorithm](https://rocm.docs.amd.com/projects/MIOpen/en/docs-6.1.0/doxygen/html/group__convolutions.html#gaca2f3b99b04393beebaee41e3d990f68)
Compare with cpu inference performance via mkldnn/zendnn

