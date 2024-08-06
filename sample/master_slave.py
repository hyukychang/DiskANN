from mpi4py import MPI
from mpi4py.MPI import Intracomm
import socket


def master(comm: Intracomm, size: int):
    tasks = range(1, size)  # Number of tasks equal to the number of slaves
    results = []

    for i in tasks:
        data = {"task_id": i, "data": i * 2}
        comm.send(data, dest=i, tag=11)
        print(f"Master sent data {data} to slave {i}")

    for i in tasks:
        result = comm.recv(source=i, tag=22)
        results.append(result)
        print(f"Master received result {result} from slave {i}")

    print("All tasks completed. Results:", results)


def slave(comm: Intracomm, rank: int):
    data = comm.recv(source=0, tag=11)
    print(f"Slave {rank} received data {data}")

    task_id = data["task_id"]
    value = data["data"]
    result = value**2  # Example computation: square the data
    host_id = socket.gethostname()
    comm.send({"task_id": task_id, "result": result}, dest=0, tag=22)
    print(f"Slave {rank} sent result {result} for task {task_id} from {host_id}")


def main():
    comm: Intracomm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    size: int = comm.Get_size()

    if rank == 0:
        print(f"Running as master with size {size}")
        master(comm, size)
    else:
        print(f"Running as slave {rank}")
        slave(comm, rank)


if __name__ == "__main__":
    main()
