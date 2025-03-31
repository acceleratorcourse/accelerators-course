# Lab4

In this lab you will write different implementations of convolution 


On Nvidia platform you need to install
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit)
* [cudnn frontend](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/frontend.html)

On Amd platform you need to install  
* [rocm](https://rocm.docs.amd.com/en/latest/)
* [MiOpen](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html)

For AMD CPU you need to install
* [ZenDNN](https://github.com/amd/ZenDNN)

For Intel CPU you need to install
* [mklidnn](https://github.com/uxlfoundation/oneDNN)


Fix CmakeLists.txt for target platform

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
sudo ./gtest/test_conv
```

## Formatting the code

All the code is formatted using `clang-format`. To format a file, use:

```shell
find . -name *.cpp | xargs clang-format-10 -style=file -i
find . -name *.hpp | xargs clang-format-10 -style=file -i 
find . -name *.h | xargs clang-format-10 -style=file -i 
```

Go to [Task doc](https://github.com/acceleratorcourse/accelerators-course/blob/main/lab4/task/TASK.md)
