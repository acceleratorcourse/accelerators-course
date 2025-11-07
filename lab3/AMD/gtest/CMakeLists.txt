find_package(GTest REQUIRED)
find_package(hip REQUIRED)
find_package(rocprim REQUIRED)
find_package(rocthrust REQUIRED)
find_package(rocblas REQUIRED)
find_package(BLAS REQUIRED)

function(add_gtest TEST_NAME TEST_CPP)
  message("Adding Test: " ${TEST_NAME} " : " ${TEST_CPP})
  add_executable(${TEST_NAME} ${TEST_CPP})

  target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
  target_include_directories(${TEST_NAME} PRIVATE ../ ../include /usr/include/mkl)

  target_link_libraries(${TEST_NAME} ${CMAKE_DL_LIBS} GTest::gtest GTest::gtest_main  ${Boost_LIBRARIES} hip::host roc::rocthrust roc::rocblas ${BLAS_LIBRARIES})
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  add_dependencies(check ${TEST_NAME})

  string(CONCAT TEST_ENVIRONMENT_VARIABLES)
  gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR} PROPERTIES ${TEST_ENVIRONMENT_VARIABLES})
endfunction()

file(GLOB TESTS *.cpp)

foreach(TEST ${TESTS})
  get_filename_component(BASE_NAME ${TEST} NAME_WE)
  add_gtest(test_${BASE_NAME} ${BASE_NAME}.cpp)
endforeach()
