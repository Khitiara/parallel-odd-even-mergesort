#include <mpi.h>
#include "timing.h"

#define DATA_LENGTH 1 << 27

int arraylen;
long long* array;    // length arraylen
long long* scratch;  // length 2 * arraylen
int mpi_rank, mpi_size;

void merge(void);

int main(int argc, char **argv) {
    return 0;
}

/**
 * Merges two arrays stored in the lower and upper halves of scratch
 * into array.
 */
void merge(void) {
    return;
}
