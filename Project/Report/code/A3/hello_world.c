#include <stdio.h>
#include <mpi.h>
int main(int argc, char** argv) {
    // Initialisation
    //int MPI_Init_thread( *argc, ***argv, MPI_THREAD_SINGLE, &provided  )
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Hello World from rank %d out from %d processes!\n",
           rank, size);
    MPI_Finalize();
    return 0;
}