#include <omp.h>  
#include <stdio.h>  
int main() {  
    #pragma omp parallel  // Code block will be executed by each thread.
    {  
        omp_set_num_threads(32); // # of threads in the code
      
        // int numThreads = omp_get_num_threads(); 

        int ID = omp_get_thread_num();  
        printf("Hello World from Thread %d!\n", ID);  
    }  
    return 0;  
}  