find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
find_package(BLAS REQUIRED)

thrust_create_target(THRUST_TARGET)

function(add_gtest TEST_NAME TEST_CPP)
  message("Adding Test: " ${TEST_NAME} " : " ${TEST_CPP})
  add_executable(${TEST_NAME} ${TEST_CPP})
  set_property(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES "90")

  target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef -g3 -O1 --save-temps)

  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef --cuda-gpu-arch=sm_90 --cuda-path=/usr/local/cuda-12.9 -O1 --save-temps)
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef -g3 -O1 --save-temps)
  endif()
	  
  target_include_directories(${TEST_NAME} PRIVATE ../ ../include /usr/include/)

  target_link_libraries(${TEST_NAME} ${CMAKE_DL_LIBS} GTest::gtest GTest::gtest_main  ${Boost_LIBRARIES} THRUST_TARGET CUDA::cublas ${BLAS_LIBRARIES} cudart_static)
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  add_dependencies(check ${TEST_NAME})

  string(CONCAT TEST_ENVIRONMENT_VARIABLES)
  gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR} PROPERTIES ${TEST_ENVIRONMENT_VARIABLES})
endfunction()

file(GLOB TESTS *.cu)

foreach(TEST ${TESTS})
  get_filename_component(BASE_NAME ${TEST} NAME_WE)
  add_gtest(test_${BASE_NAME} ${BASE_NAME}.cu)
endforeach()
