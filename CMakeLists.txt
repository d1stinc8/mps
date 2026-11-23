cmake_minimum_required(VERSION 3.20)
project(mps CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

include(FetchContent)

# Eigen: SVD and dense linear algebra.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(eigen)

# GoogleTest.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Core library: header-only, interface target.
add_library(mps_core INTERFACE)
target_include_directories(mps_core INTERFACE src)
target_link_libraries(mps_core INTERFACE Eigen3::Eigen)
target_compile_options(mps_core INTERFACE -Wall -Wextra -Wpedantic)

# Debug: catch memory bugs early.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(mps_core INTERFACE -fsanitize=address,undefined)
  target_link_options(mps_core INTERFACE -fsanitize=address,undefined)
endif()

add_executable(mps_main src/main.cpp)
target_link_libraries(mps_main PRIVATE mps_core)

enable_testing()
add_executable(test_gates tests/test_gates.cpp)
target_link_libraries(test_gates PRIVATE mps_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_gates)
add_executable(test_2q tests/test_2q.cpp)
target_link_libraries(test_2q PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_2q)
add_executable(test_canonical tests/test_canonical.cpp)
target_link_libraries(test_canonical PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_canonical)
add_executable(test_exact tests/test_exact.cpp)
target_link_libraries(test_exact PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_exact)
add_executable(test_swap tests/test_swap.cpp)
target_link_libraries(test_swap PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_swap)
add_executable(test_circuits tests/test_circuits.cpp)
target_link_libraries(test_circuits PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_circuits)
add_executable(test_sampling tests/test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_sampling)
add_executable(test_convergence tests/test_convergence.cpp)
target_link_libraries(test_convergence PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_convergence)
add_executable(convergence bench/convergence.cpp)
target_link_libraries(convergence PRIVATE mps_core)
add_executable(test_rydberg tests/test_rydberg.cpp)
target_link_libraries(test_rydberg PRIVATE mps_core GTest::gtest_main)
gtest_discover_tests(test_rydberg)

add_executable(z2_sweep bench/z2_sweep.cpp)
target_link_libraries(z2_sweep PRIVATE mps_core)

add_executable(quench_entropy bench/quench_entropy.cpp)
target_link_libraries(quench_entropy PRIVATE mps_core)

# Python extension. Skipped in Debug: ASan cannot be loaded by a stock Python.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(PYBIND11_FINDPYTHON ON)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(_mps src/bindings.cpp)
  target_link_libraries(_mps PRIVATE mps_core)
  set_target_properties(_mps PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/mps)
endif()
