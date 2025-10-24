cmake_minimum_required(VERSION 3.16.3)

project(lab2 LANGUAGES HIP CXX)

set(CMAKE_HIP_COMPILER nvcc)
set(CMAKE_GPU_RUNTIME CUDA)
set(CMAKE_HIP_ARCHITECTURES 90)

include(CTest)

message(STATUS "HIP_PATH: ${HIP_PATH}")
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

add_subdirectory(gtest)

