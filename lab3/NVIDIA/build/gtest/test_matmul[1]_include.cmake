if(EXISTS "/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul")
  if(NOT EXISTS "/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul[1]_tests.cmake" OR
     NOT "/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul[1]_tests.cmake" IS_NEWER_THAN "/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul" OR
     NOT "/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul[1]_tests.cmake" IS_NEWER_THAN "${CMAKE_CURRENT_LIST_FILE}")
    include("/snap/cmake/1487/share/cmake-4.1/Modules/GoogleTestAddTests.cmake")
    gtest_discover_tests_impl(
      TEST_EXECUTABLE [==[/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul]==]
      TEST_EXECUTOR [==[]==]
      TEST_WORKING_DIR [==[/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/]==]
      TEST_EXTRA_ARGS [==[]==]
      TEST_PROPERTIES [==[]==]
      TEST_PREFIX [==[]==]
      TEST_SUFFIX [==[]==]
      TEST_FILTER [==[]==]
      NO_PRETTY_TYPES [==[FALSE]==]
      NO_PRETTY_VALUES [==[FALSE]==]
      TEST_LIST [==[test_matmul_TESTS]==]
      CTEST_FILE [==[/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul[1]_tests.cmake]==]
      TEST_DISCOVERY_TIMEOUT [==[300]==]
      TEST_DISCOVERY_EXTRA_ARGS [==[]==]
      TEST_XML_OUTPUT_DIR [==[]==]
    )
  endif()
  include("/home/kikimych/kikimych_backup/workspace/itmo/lab3/NVIDIA/build/gtest/test_matmul[1]_tests.cmake")
else()
  add_test(test_matmul_NOT_BUILT test_matmul_NOT_BUILT)
endif()
