cmake_minimum_required(VERSION 3.27)
project(paral)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -fopenmp")
set(CMAKE_CXX_STANDARD 23)

add_executable(paral task2.cpp
        task4_1.cpp
        task5.cpp)
