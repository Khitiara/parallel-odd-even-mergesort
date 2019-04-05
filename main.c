#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include <string.h>

#define DATA_LENGTH (1ul << 27u)

size_t arraylen;
long long *array;    // length arraylen
long long *scratch;  // length 2 * arraylen
int mpi_rank, mpi_size;

int comp(const void *elem1, const void *elem2);

int merge_lower(void);

int merge_upper(void);

int merges(void);

int check_sorted(void);

void load(char *fpath);

int main(int argc, char **argv) {
    unsigned long long start_cycles = 0;
    unsigned long long end_cycles = 0;
    double time;
    int iterations = 0;
    int actually_sorted;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 2) {
        printf("Usage: %s <input file>\n", argv[0]);
        exit(1);
    }

    arraylen = DATA_LENGTH / mpi_size;

    array = calloc(arraylen, sizeof(long long));
    scratch = calloc(arraylen * 2, sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    start_cycles = GetTimeBase();
    load(argv[1]);
    MPI_Barrier(MPI_COMM_WORLD);
    end_cycles = GetTimeBase();
    time = (end_cycles - start_cycles) / g_processor_frequency;
    if (mpi_rank == 0) {
        printf("Loaded input data in %lf seconds.\n", time);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_cycles = GetTimeBase();
    qsort(array, arraylen, sizeof(long long), comp);
    while(merges()) {
        ++iterations;
    }
    ++iterations; // Last iteration
    MPI_Barrier(MPI_COMM_WORLD);
    end_cycles = GetTimeBase();
    time = (end_cycles - start_cycles) / g_processor_frequency;
    if (mpi_rank == 0) {
        printf("Sorted %lu elements in %lf seconds with %d iterations.\n", DATA_LENGTH, time, iterations);
    }

    actually_sorted = check_sorted();
    if (mpi_rank == 0) {
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
    long long *dst = array;
    long long *a = scratch;
    long long *b = scratch + arraylen;
    int i, order_changed = 0;
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
    long long *dst = array;
    long long *a = scratch;
    long long *b = scratch + arraylen;
    int i, order_changed = 0;
    for (i = 0; i < arraylen; ++i) {
        if (*a >= *b) {
            *dst++ = *a++;
        } else {
            *dst++ = *b++;
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

#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

int merges(void) {
    int parity = mpi_rank % 2, changed = 0, anychanges;
    MPI_Request reqs[2];

    // Even merge
    memcpy(scratch, array, arraylen * sizeof(long long));
    if (0 == parity) {
        MPI_Isend(array, arraylen, MPI_LONG_LONG_INT, mpi_rank + 1, 0, MPI_COMM_WORLD, reqs);
        MPI_Irecv(scratch + arraylen, arraylen, MPI_LONG_LONG_INT, mpi_rank + 1, 1, MPI_COMM_WORLD, reqs + 1);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        changed |= merge_lower();
    } else {
        MPI_Isend(array, arraylen, MPI_LONG_LONG_INT, mpi_rank - 1, 0, MPI_COMM_WORLD, reqs);
        MPI_Irecv(scratch + arraylen, arraylen, MPI_LONG_LONG_INT, mpi_rank - 1, 1, MPI_COMM_WORLD, reqs + 1);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        changed |= merge_upper();
    }

    // Odd merge - first and last ranks do nothing
    if (0 != mpi_rank && mpi_size - 1 != mpi_rank) {
        if (1 == parity) {
            MPI_Isend(array, arraylen, MPI_LONG_LONG_INT, mpi_rank + 1, 0, MPI_COMM_WORLD, reqs);
            MPI_Irecv(scratch + arraylen, arraylen, MPI_LONG_LONG_INT, mpi_rank + 1, 1, MPI_COMM_WORLD, reqs + 1);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            changed |= merge_lower();
        } else {
            MPI_Isend(array, arraylen, MPI_LONG_LONG_INT, mpi_rank - 1, 0, MPI_COMM_WORLD, reqs);
            MPI_Irecv(scratch + arraylen, arraylen, MPI_LONG_LONG_INT, mpi_rank - 1, 1, MPI_COMM_WORLD, reqs + 1);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            changed |= merge_upper();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&changed, &anychanges, 1, MPI_LONG_LONG_INT, MPI_LOR, MPI_COMM_WORLD);
    return anychanges;
}

#pragma clang diagnostic pop

void load(char *fpath) {
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
