cmake_minimum_required(VERSION 3.27)
project(paral)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -fopenmp")
set(CMAKE_CXX_STANDARD 23)

include_directories(/usr/include/mpi/)

set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

add_executable(paral task15.cpp)
