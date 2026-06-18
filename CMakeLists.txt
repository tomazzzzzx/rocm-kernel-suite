cmake_minimum_required(VERSION 3.20)
project(rocm-kernel-suite LANGUAGES CXX)

find_package(hip REQUIRED)

add_library(kernels INTERFACE)
target_include_directories(kernels INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

enable_testing()
add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm hip::device)
add_test(NAME test_gemm COMMAND test_gemm)
