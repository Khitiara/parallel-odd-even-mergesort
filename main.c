#include <mpi.h>
#include "timing.h"

#define DATA_LENGTH 1 << 27

int arraylen;
long long *array;    // length arraylen
long long *scratch;  // length 2 * arraylen
int mpi_rank, mpi_size;

int merge_lower(void);
int merge_upper(void);

void load(char *fpath);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    return 0;
}

/**
 * Merges two arrays stored in the lower and upper halves of scratch
 * and stores the lower half into array.
 */
int merge_lower(void) {
    long long* dst = array;
    long long* a = scratch;
    long long* b = scratch + arraylen;
    int i, order_changed = 0;
    for(i = 0; i < arraylen; ++i) {
        if(*a < *b) {
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
int merge_lower(void) {
    long long* dst = array;
    long long* a = scratch;
    long long* b = scratch + arraylen;
    int i, order_changed = 0;
    for(i = 0; i < arraylen; ++i) {
        if(*a >= *b) {
            *dst++ = *a++;
        } else {
            *dst++ = *b++;
            order_changed = 1;
        }
    }
    return order_changed;
}

void load(char *fpath) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, fpath, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at_all(fh, mpi_rank * arraylen, array, arraylen, MPI_LONG_LONG_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}
