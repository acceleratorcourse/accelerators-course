# Lab3

In third lab you will learn to boost performance of gpu programs

Best example is matrix multiplication 

On Nvidia platform you need to install
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit)
* [cuiblas](https://developer.nvidia.com/cublas)

On Amd platform you need to install  
* [rocm](https://rocm.docs.amd.com/en/latest/)
* [rocblas](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html)

For CPU part of the test you need to install 
* [mkl](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
* [openblas](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/install.md)


## Building tests on AMD platform

First, create a build directory:

```shell
mkdir build; cd build;
```

Next, configure CMake

```shell
cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DHIP_CXX_COMPILER=/opt/rocm/bin/hipcc ..
```
Then, build tests

```shell
make
```
## Running the tests
To run a test, use the following code:

```shell
sudo ./gtest/test_matmul
```

## Building tests on NVIDIA platform

First, create a build directory:

```shell
mkdir build; cd build;
```

Next, set environment variables for MKL library
```shell
source /opt/intel/oneapi/setvars.sh
```
Next, configure CMake

```shell
NVCC_PREPEND_FLAGS=' -ccbin clang-14' cmake  ..
```
Then, build tests

```shell
NVCC_PREPEND_FLAGS=' -ccbin clang-14' make
```
## Running the tests
To run a test, use the following code:

```shell
./gtest/test_matmul
```

## Formatting the code

All the code is formatted using `clang-format`. To format a file, use:

```shell
find . -name *.cpp | xargs clang-format-10 -style=file -i
find . -name *.hpp | xargs clang-format-10 -style=file -i 
find . -name *.h   | xargs clang-format-10 -style=file -i 
find . -name *.cu  | xargs clang-format-10 -style=file -i
```

Go to [Task doc](https://github.com/acceleratorcourse/accelerators-course/blob/main/lab3/task/TASK.md)
