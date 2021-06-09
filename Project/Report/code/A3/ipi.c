
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define MINITER 16
#define MAXITER 1 << 23

int main(int argc, char* argv[])
{
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(SEED + size + rank*32); // Important: Multiply SEED by "rank" when you introduce MPI!
    // Calculate PI following a Monte Carlo method
    for(int iters = MINITER; iters <= MAXITER; iters *= 2){
		for(int rep=0; rep<10; rep++){
            MPI_Barrier(MPI_COMM_WORLD);
            int count = 0;
            double x, y, z, pi;

            double tstart = MPI_Wtime();
            // Calculate PI following a Monte Carlo method

            if (rank!=0){
                for (int i = 0; i < iters; i++)
                {
                    // Generate random (X,Y) points
                    x = (double)random() / (double)RAND_MAX;
                    y = (double)random() / (double)RAND_MAX;
                    z = sqrt((x*x) + (y*y));
                    
                    // Check if point is in unit circle
                    count += (z <= 1.0);
                }
                MPI_Send(&count, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
            }
            else{
                MPI_Request requests[size-1];
                int counts[size];
                for (int i=1; i<size; i++){
                    MPI_Irecv(&counts[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, &requests[i-1]);
                }

                for (int i = 0; i < iters; i++){
                    // Generate random (X,Y) points
                    x = (double)random() / (double)RAND_MAX;
                    y = (double)random() / (double)RAND_MAX;
                    z = sqrt((x*x) + (y*y));
                    
                    // Check if point is in unit circle
                    count += (z <= 1.0);
                }

                MPI_Waitall(size-1, requests, MPI_STATUS_IGNORE);

                for (int i=1; i<size; i++){
                    count += counts[i];
                }

                pi = ((double)count / (((double)iters)*size) ) * 4.0;
                double tend = MPI_Wtime();
                double dt = tend - tstart;
                printf("proc=%d, N=%d, val=%f, time=%fms\n", size, iters, pi, 1000*dt);
            }
        }
    }

    // Estimate Pi and display the result
    MPI_Finalize();
    return 0;
}
