MPI_Bcast(currA, locsize*locsize, MPI_DOUBLE, sendercol, rowcomm);

//compute
//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, locsize, locsize, locsize, 1, currA, locsize, locB, locsize, 1.0, locC, locsize);
matmul(locsize, currA, locB, locC);

MPI_Request request;
MPI_Irecv(buffB, locsize*locsize, MPI_DOUBLE, down, col, cartcom, &request);
MPI_Send(locB, locsize*locsize, MPI_DOUBLE, up, col, cartcom);
MPI_Status status;
MPI_Wait(&request, &status);
double* temp = locB;
locB = buffB;
buffB = temp;