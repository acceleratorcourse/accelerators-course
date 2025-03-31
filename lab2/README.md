# Lab2

In second lab you need to learn how to deal with both CPU C++ standart library and GPU standart
library

You can find C++ standart library documentation at [C++ std lib](https://en.cppreference.com/w/)
Nvidia's GPU standart library docs at [thrust docs](https://nvidia.github.io/cccl/thrust/) and
Amd's implementation at [Rocthrust repo](https://github.com/ROCm/rocThrust/tree/develop).


You can write excersises on Nvidia platform using 
* [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/)
* [CUDA] (deprecated)

On Amd platform you need to write on 
* [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/)

## Building tests on AMD platform

To run tests, you must first install these prerequisites:

* A [ROCm](https://rocm.docs.amd.com/)-enabled platform
* HIP (HIP and HCC libraries and header files)
* [Half](http://half.sourceforge.net/): IEEE 754-based, half-precision floating-point library
* [Google test](https://google.github.io/googletest/)
* [clang-format](https://clang.llvm.org/docs/ClangFormat.html)


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

You can run all tests using the 'check' target:

` make check `

To run a single test, use the following code:

```shell
sudo ./gtest/test_group_by
sudo ./gtest/test_prefix_sum
...
```

## Formatting the code

All the code is formatted using `clang-format`. To format a file, use:

```shell
find . -name *.cpp | xargs clang-format-10 -style=file -i
find . -name *.hpp | xargs clang-format-10 -style=file -i 
find . -name *.h | xargs clang-format-10 -style=file -i 
```

Go to [Task doc](task/Task.md)
