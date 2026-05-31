cmake_minimum_required(VERSION 3.20)
project(ProgettoParallelComputing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(ProgettoParallelComputing main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ProgettoParallelComputing PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(ProgettoParallelComputing PRIVATE -O3)