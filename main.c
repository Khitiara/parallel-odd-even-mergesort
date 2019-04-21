#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include <string.h>

#define DATA_LENGTH (1ul << 27u)
#define UP_TAG 0
#define DOWN_TAG 1

size_t g_arraylen;
long long *array;    // length arraylen
long long *scratch;  // length 2 * arraylen
int g_mpi_rank, mpi_size;

int comp(const void *elem1, const void *elem2);

int merge_lower(void);

int merge_upper(void);

int merges(void);

int check_sorted(void);

void load(char *fpath);

int main(int argc, char **argv) {
    unsigned long long start_cycles = 0;
    unsigned long long end_cycles = 0;
    double merge_time, serial_time;
    int iterations = 0;
    int actually_sorted;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);

    if (argc != 2) {
        printf("Usage: %s <input file>\n", argv[0]);
        exit(1);
    }

    const size_t arraylen = g_arraylen = DATA_LENGTH / mpi_size;

    array = calloc(arraylen, sizeof(long long));
    scratch = calloc(arraylen * 2, sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    start_cycles = GetTimeBase();
    load(argv[1]);
    MPI_Barrier(MPI_COMM_WORLD);
    end_cycles = GetTimeBase();
    serial_time = (end_cycles - start_cycles) / g_processor_frequency;
    if (g_mpi_rank == 0) {
        printf("Loaded input data in %lf seconds.\n", serial_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_cycles = GetTimeBase();
    qsort(array, arraylen, sizeof(long long), comp);
    MPI_Barrier(MPI_COMM_WORLD);
    end_cycles = GetTimeBase();
    serial_time = (end_cycles - start_cycles) / g_processor_frequency;
    if(g_mpi_rank == 0) {
        puts("Serial sort complete");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start_cycles = GetTimeBase();
    while(merges()) {
        ++iterations;
    }
    ++iterations; // Last iteration
    MPI_Barrier(MPI_COMM_WORLD);
    end_cycles = GetTimeBase();
    merge_time = (end_cycles - start_cycles) / g_processor_frequency;
    if (g_mpi_rank == 0) {
        printf("Computation statistics:\n"
               "            Rank Count: %d\n"
               "        Total Elements: %lu\n"
               "         Elements/Rank: %lu\n"
               "            Iterations: %d\n"
               "   Serial Run Time (s): %lf\n"
               "    Merge Run time (s): %lf\n"
               "    Time/Iteration (s): %lf\n",
            mpi_size,
            DATA_LENGTH,
            arraylen,
            iterations,
            serial_time,
            merge_time,
            merge_time / iterations);
    }

    actually_sorted = check_sorted();
    if (g_mpi_rank == 0) {
        printf("Sort check: %s\n", actually_sorted ? "passed" : "failed");
    }

    free(array);
    free(scratch);

    MPI_Finalize();

    return 0;
}

/**
 * Merges two arrays stored in the lower and upper halves of scratch
 * and stores the lower half into array.
 */
int merge_lower(void) {
    const size_t arraylen = g_arraylen;
    long long *restrict dst = array;
    const long long *a = scratch;
    const long long *b = scratch + arraylen;
    size_t i;
    int order_changed = 0;
    for (i = 0; i < arraylen; ++i) {
        if (*a < *b) {
            *dst++ = *a++;
        } else {
            *dst++ = *b++;
            order_changed = 1;
        }
    }
    return order_changed;
}

/**
 * Merges two arrays stored in the lower and upper halves of scratch
 * and stores the upper half into array.
 */
int merge_upper(void) {
    const size_t arraylen = g_arraylen;
    long long *restrict dst = array + arraylen;
    const long long *a = scratch + arraylen;
    const long long *b = scratch + arraylen + arraylen;
    size_t i;
    int order_changed = 0;
    for (i = 0; i < arraylen; ++i) {
        if (a[-1] >= b[-1]) {
            *--dst = *--a;
        } else {
            *--dst = *--b;
            order_changed = 1;
        }
    }
    return order_changed;
}


int comp(const void *elem1, const void *elem2) {
    long long f = *((long long *) elem1);
    long long s = *((long long *) elem2);
    if (f > s) return 1;
    if (f < s) return -1;
    return 0;
}

int exchange_lower(const int mpi_rank, const size_t arraylen) {
    int changed;
    MPI_Request reqs[2];
    MPI_Irecv(scratch + arraylen, arraylen, MPI_LONG_LONG_INT, mpi_rank + 1, DOWN_TAG, MPI_COMM_WORLD, reqs + 1);
    MPI_Isend(array, arraylen, MPI_LONG_LONG_INT, mpi_rank + 1, UP_TAG, MPI_COMM_WORLD, reqs);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    #if USE_OPTIMIZED_MERGE
    if (array[arraylen - 1] <= scratch[arraylen]) {
        // if the max of this rank is less than or equal to the min of the other rank,
        // the we are already sorted.
        changed = 0;
    } else if (array[0] >= scratch[arraylen + arraylen - 1]) {
        // If the min of this rank is greater than or
        // equal to the max of the other rank, then we must be swqpped
        memcpy(array, scratch + arraylen, arraylen * sizeof(long long));
        changed = 1;
    } else {
        memcpy(scratch, array, arraylen * sizeof(long long));
        changed = merge_lower();
    }
    #else
    memcpy(scratch, array, arraylen * sizeof(long long));
    changed = merge_lower();
    #endif
    return changed;
}

int exchange_upper(const int mpi_rank, const size_t arraylen) {
    int changed;
    MPI_Request reqs[2];
    MPI_Irecv(scratch + arraylen, arraylen, MPI_LONG_LONG_INT, mpi_rank - 1, UP_TAG, MPI_COMM_WORLD, reqs + 1);
    MPI_Isend(array, arraylen, MPI_LONG_LONG_INT, mpi_rank - 1, DOWN_TAG, MPI_COMM_WORLD, reqs);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    #if USE_OPTIMIZED_MERGE
    if (array[arraylen - 1] <= scratch[arraylen]) {
        // we must swap
        memcpy(array, scratch + arraylen, arraylen * sizeof(long long));
        changed = 1;
    } else if (array[0] >= scratch[arraylen + arraylen - 1]) {
        //do nothing
        changed = 0;
    } else {
        memcpy(scratch, array, arraylen * sizeof(long long));
        changed = merge_upper();
    }
    #else
    memcpy(scratch, array, arraylen * sizeof(long long));
    changed = merge_upper();
    #endif
    return changed;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

int merges(void) {
    const size_t arraylen = g_arraylen;
    const int mpi_rank = g_mpi_rank;
    int parity = mpi_rank % 2, changed = 0, anychanges;

    // Even merge
    if (0 == parity) {
        changed |= exchange_lower(mpi_rank, arraylen);
    } else {
        changed |= exchange_upper(mpi_rank, arraylen);
    }

    // Odd merge - first and last ranks do nothing
    if (0 != mpi_rank && mpi_size - 1 != mpi_rank) {
        if (1 == parity) {
            changed |= exchange_lower(mpi_rank, arraylen);
        } else {
            changed |= exchange_upper(mpi_rank, arraylen);
        }
    }
    MPI_Allreduce(&changed, &anychanges, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return anychanges;
}

#pragma clang diagnostic pop

void load(char *fpath) {
    const size_t arraylen = g_arraylen;
    const int mpi_rank = g_mpi_rank;
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, fpath, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at_all(fh, mpi_rank * arraylen * sizeof(long long), array, arraylen, MPI_LONG_LONG_INT,
                         MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

/**
 * Checks that the final array is actually sorted.
 */
int check_sorted(void) {
    const size_t arraylen = g_arraylen;
    const int mpi_rank = g_mpi_rank;
    MPI_Request req[2];
    long long max_from_prev_rank;
    int sorted = 1, finalsorted = 1;
    size_t i;
    // receive the maximum element from the previous rank
    if (mpi_rank != 0) {
        MPI_Irecv(&max_from_prev_rank, 1, MPI_LONG_LONG_INT, mpi_rank - 1, 0, MPI_COMM_WORLD, &req[0]);
    }
    // check our own rank
    for (i = 1; i < arraylen; ++i) {
        if (array[i] < array[i - 1]) {
            sorted = 0;
            break;
        }
    }
    // send out the maximum element
    if (mpi_rank != mpi_size - 1) {
        MPI_Isend(&array[i - 1], 1, MPI_LONG_LONG_INT, mpi_rank + 1, 0, MPI_COMM_WORLD, &req[1]);
    }
    if (mpi_rank != 0) {
        // compare max from the prev rank
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        if (array[0] < max_from_prev_rank) {
            sorted = 0;
        }
    }
    MPI_Reduce(&sorted, &finalsorted, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    return finalsorted;
}
