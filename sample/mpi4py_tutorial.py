from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
host_name = socket.gethostname()
print(f"execute from {host_name} : {rank}")
