#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <limits.h>

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char hostname[100];
    if (gethostname(hostname, sizeof(hostname)) == 0)
    {
        std::cout << "get host name" << std::endl;
    }
    else
    {
        std::cerr << "failed to get host name" << std::endl;
    }
    printf("Hello world from process %d of %d in host name %s\n", rank, size, hostname);

    MPI_Finalize();
    return 0;
}
