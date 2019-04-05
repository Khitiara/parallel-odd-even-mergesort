cmake_minimum_required (VERSION 3.0)
find_program(MPI_WRAPPER NAMES mpicc mpixlc)
set(CMAKE_C_COMPILER ${MPI_WRAPPER})

project(parallel-sort)

option(BGQ "Running on BGQ?" OFF)

if(BGQ)
    add_definitions(-DBGQ=1)
else(BGQ)
    add_definitions(-DBGQ=0)
endif(BGQ)

add_executable(sort main.c)
