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

int merge_lower(void);

int merge_upper(void);

int merges(void);

void load(char *fpath);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 2) {
        printf("Usage: %s <output file>\n", argv[0]);
        exit(1);
    }

    arraylen = DATA_LENGTH / mpi_size;

    load(argv[1]);

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
    qsort(array, arraylen, sizeof(long long), comp);

    // Odd merge
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

    // Even merge
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
    MPI_File_read_at_all(fh, mpi_rank * arraylen * sizeof(long long int), array, arraylen, MPI_LONG_LONG_INT,
                         MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}
