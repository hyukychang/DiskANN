#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <limits.h>

// Master function
void master(int num_tasks)
{
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_workers = world_size - 1;
    int tasks_sent = 0;
    int tasks_completed = 0;
    int terminator = 0;

    // Distribute tasks
    for (int rank = 1; rank <= num_workers && tasks_sent < num_tasks; ++rank)
    {
        MPI_Send(&tasks_sent, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
        std::cout << "Master sent task " << tasks_sent << " to worker " << rank << std::endl;
        ++tasks_sent;
    }

    // Collect results and send new tasks
    while (tasks_completed < num_tasks)
    {
        int task_result;
        MPI_Status status;

        // Receive result from any worker
        MPI_Recv(&task_result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if (task_result == -1)
        {
            ++terminator;
            continue;
        }

        std::cout << "Master received result " << task_result << " from worker " << status.MPI_SOURCE << std::endl;
        ++tasks_completed;
        std::cout << "complete task num : " << tasks_completed << std::endl;

        // Send new task if available
        if (tasks_sent < num_tasks)
        {
            MPI_Send(&tasks_sent, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            std::cout << "Master sent task " << tasks_sent << " to worker " << status.MPI_SOURCE << std::endl;
            ++tasks_sent;
        }
        else
        {
            // No more tasks to send, send termination signal
            std::cout << "no more task to send, sending termination sinal" << std::endl;
            int terminate = -1;
            MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }
    }

    // Finalize all workers
    for (int rank = 1; rank <= num_workers - terminator; ++rank)
    {
        std::cout << "finalize with rank : " << rank << std::endl;
        int task_result;
        MPI_Recv(&task_result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    std::cout << "Finalize finish" << std::endl;
}
// Worker function
void worker()
{
    int task;
    MPI_Status status;
    char hostname[100];

    if (gethostname(hostname, sizeof(hostname)) == 0)
    {
        std::cout << "get host name" << std::endl;
    }
    else
    {
        std::cerr << "failed to get host name" << std::endl;
    }

    while (true)
    {
        // Receive task from master
        MPI_Recv(&task, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std::cout << hostname << " receive task(" << task << ") from master" << std::endl;

        // Check for termination signal
        if (task == -1)
        {
            std::cout << "get terminational signal from master" << std::endl;
            int result = task;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            break;
        }

        // Process the task (for simplicity, just return the task number)
        int result = task;

        // Send result back to master
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int num_tasks = 10; // Total number of tasks to process

    if (world_rank == 0)
    {
        master(num_tasks);
    }
    else
    {
        worker();
    }

    MPI_Finalize();
    return 0;
}