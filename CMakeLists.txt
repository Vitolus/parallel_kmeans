cmake_minimum_required(VERSION 3.22.1)
project(parallel_kmeans)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(parallel_kmeans main.cpp
        k_means.cpp
        k_means.h
)

# Link against OpenMP
if(OPENMP_FOUND)
    target_link_libraries(parallel_kmeans PUBLIC OpenMP::OpenMP_CXX)
endif()