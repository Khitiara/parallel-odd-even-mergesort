cmake_minimum_required (VERSION 2.6)
find_program(MPI_WRAPPER NAMES mpicc mpixlc)
set(CMAKE_C_COMPILER ${MPI_WRAPPER})

project(parallel-sort)

add_executable(sort main.c)
