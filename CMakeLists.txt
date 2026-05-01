cmake_minimum_required(VERSION 3.16)

project(skdtree_benchamrk)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS "-mbmi2 -O3 -DNDEBUG -march=native")

set(BOOST_ROOT "PATH TO boost_1_79_0")
set(Boost_INCLUDE_DIR "PATH TO boost_1_79_0")
set(Boost_LIBRARY_DIR "PATH TO boost_1_79_0/stage/lib")

SET(EXECUTABLE_OUTPUT_PATH "bin")

# check dependencies
find_package(Boost 1.79.0 REQUIRED COMPONENTS program_options system filesystem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${Boost_INCLUDE_DIR})


#QUERIES EXECUTABLES
# NOTE:
# You can manually add a new executable according to the dimensionality
# of the dataset you want to run for the queries
# This file includes the baseline executables for the datasets
# contained in the "data/dataset" folder, i.e., for d = {2,4,6,8}

add_executable(bench2d bench/bench.cpp)
target_link_libraries(bench2d Boost::program_options -larmadillo)

add_executable(bench4d bench/bench.cpp)
target_link_libraries(bench4d Boost::program_options -larmadillo)
target_compile_definitions(bench4d PUBLIC BENCH_DIM=4)

add_executable(bench6d bench/bench.cpp)
target_link_libraries(bench6d Boost::program_options -larmadillo)
target_compile_definitions(bench6d PUBLIC BENCH_DIM=6)

add_executable(bench8d bench/bench.cpp)
target_link_libraries(bench8d  Boost::program_options -larmadillo)
target_compile_definitions(bench8d PUBLIC BENCH_DIM=8)

# MIXED WORKLOAD EXECUTABLES
# NOTE:
# You can manually add a new executable according to the dimensionality
# of the dataset you want to run for the mixed workloads
# This file includes the baseline executables for the datasets
# contained in the "data/dataset" folder, i.e., for d = {2,4,6,8}

add_executable(bench2d_mixed bench/bench_mixed_workload.cpp)
target_link_libraries(bench2d_mixed Boost::program_options -larmadillo)

add_executable(bench4d_mixed bench/bench_mixed_workload.cpp)
target_link_libraries(bench4d_mixed Boost::program_options -larmadillo)
target_compile_definitions(bench4d_mixed PUBLIC BENCH_DIM=4)

add_executable(bench6d_mixed bench/bench_mixed_workload.cpp)
target_link_libraries(bench6d_mixed Boost::program_options -larmadillo)
target_compile_definitions(bench6d_mixed PUBLIC BENCH_DIM=6)

add_executable(bench8d_mixed bench/bench_mixed_workload.cpp)
target_link_libraries(bench8d_mixed Boost::program_options -larmadillo)
target_compile_definitions(bench8d_mixed PUBLIC BENCH_DIM=8)