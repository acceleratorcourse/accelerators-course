cmake_minimum_required(VERSION 3.16.3)

project(sem1 LANGUAGES CUDA CXX)

include(CTest)


add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)


add_subdirectory(gtest)

