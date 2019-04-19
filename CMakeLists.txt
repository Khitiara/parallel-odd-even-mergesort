cmake_minimum_required (VERSION 3.0)
find_program(MPI_WRAPPER NAMES mpicc mpixlc)
set(CMAKE_C_COMPILER ${MPI_WRAPPER})

project(parallel-sort)

option(BGQ "Running on BGQ?" OFF)
if(BGQ)
    set(OPTIMIZATION_FLAG -O5)
else(BGQ)
    set(OPTIMIZATION_FLAG -O3)
endif(BGQ)

configure_file(timing.h.in timing.h)

include_directories(${CMAKE_BINARY_DIR})
add_executable(sort-default main.c)
add_executable(sort-optmerge main.c)

target_compile_definitions(sort-default PRIVATE USE_OPTIMIZED_MERGE=0)
target_compile_definitions(sort-optmerge PRIVATE USE_OPTIMIZED_MERGE=1)

target_compile_options(sort-default PRIVATE ${OPTIMIZATION_FLAG})
target_compile_options(sort-optmerge PRIVATE ${OPTIMIZATION_FLAG})
