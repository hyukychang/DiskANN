from mpi4py import MPI
from mpi4py.MPI import Intracomm
import socket
import os

MASTER_RANK = 0

MASTER_TO_WORKER = 11
WORKER_TO_MASTER = 22

DISKANN_PATH = "/home/hyuk/DiskANN"
BUILD_PATH = f"{DISKANN_PATH}/build"
APP_PATH = f"{BUILD_PATH}/apps"
SAMPLE_PATH = f"{DISKANN_PATH}/sample_mpi"

DATA_PATH = f"{DISKANN_PATH}/build/data/sift"

INDEX_PATH = f"{DATA_PATH}/base"
INDEX_PREFIX = "disk_index_sift_base_R32_L50_A1.2_mem.index_tempFiles_subshard-"
INDEX_SUFFIX = "_mem.index_custom"

QUERY_FILE = f"{DATA_PATH}/sift_query.fbin"
GT_FILE = f"{DATA_PATH}/sift_query_base_gt_100"

RESULT_PREFIX = f"{SAMPLE_PATH}/result/mem-"

# for i in range(3):
#     index_file = f"{INDEX_PREFIX}{i}{INDEX_SUFFIX}"
#     index_path_prefix = f"{INDEX_PATH}/{index_file}"
#     result_path = f"{RESULT_PREFIX}{i}"
#     search_memory_index = f"{APP_PATH}/search_memory_index"
#     command = f"{search_memory_index} --data_type float --dist_fn l2 --index_path_prefix {index_path_prefix} --query_file {QUERY_FILE} -K 10 -L 10 20 30 40 50 100 1000 10000 --result_path {result_path} --gt_file {GT_FILE}"
#     print(command)

"""
./apps/search_memory_index \
--data_type float \
--dist_fn l2 \
--index_path_prefix data/sift/base/disk_index_sift_base_R32_L50_A1.2_mem.index_tempFiles_subshard-0_mem.index_custom \
--query_file data/sift/sift_query.fbin \
-K 10 \
-L 10 20 30 40 50 100 1000 10000 \
--result_path data/sift/base/result/mem-0 \
--gt_file data/sift/base/sift_query_base_gt_100
"""


def build_command(data: int) -> str:
    index_file = f"{INDEX_PREFIX}{data}{INDEX_SUFFIX}"
    index_path_prefix = f"{INDEX_PATH}/{index_file}"
    result_path = f"{RESULT_PREFIX}{data}"
    search_memory_index = f"{APP_PATH}/search_memory_index"
    command = f"{search_memory_index} --data_type float --dist_fn l2 --index_path_prefix {index_path_prefix} --query_file {QUERY_FILE} -K 10 -L 10 20 30 40 50 100 1000 10000 --result_path {result_path} --gt_file {GT_FILE}"
    return command


def work(task_id: int, data: int) -> int:
    host_id = socket.gethostname()
    print(f"{host_id} process for task_id : {task_id}")
    command = build_command(data)
    print(command)
    os.system(f"{command} > mem_search.txt")
    return data**2


def generate_tasks(size: int) -> list:
    return [{"task_id": i + 1, "data": i, "target_rank": i} for i in range(size)]


def _get_task_id(task: dict) -> int:
    return task["task_id"]


def _get_target_rank(task: dict) -> int:
    return task["target_rank"]


def _get_task_data(task: dict) -> int:
    return task.get("data", 0)


def _get_result_from_response(response_data: dict) -> int:
    return response_data["result"]


def master_with_worker(comm: Intracomm, size: int):
    # tasks = range(1, size)  # Number of tasks equal to the number of slaves
    results = []
    tasks = generate_tasks(size=size)

    for task in tasks:
        destination = _get_target_rank(task)
        if destination == MASTER_RANK:
            # don't have to send task to self
            continue
        else:
            comm.send(task, dest=destination, tag=MASTER_TO_WORKER)
            print(f"Master sent data {task} to slave {task['task_id']}")

    for task in tasks:
        destination = _get_target_rank(task)
        if destination == MASTER_RANK:
            result = work(task_id=_get_task_id(task), data=_get_task_data(task))
            results.append(result)
        else:
            response_data = comm.recv(source=destination, tag=WORKER_TO_MASTER)
            results.append(_get_result_from_response(response_data))
            print(f"Master received result {result} from slave {task}")

    print("All tasks completed. Results:", results)


def _generate_response(task_id: int, result: int) -> dict:
    return {"task_id": task_id, "result": result}


def slave(comm: Intracomm, rank: int):
    task = comm.recv(source=MASTER_RANK, tag=MASTER_TO_WORKER)
    task_id = _get_task_id(task)
    data = _get_task_data(task)
    result = work(task_id, data)
    response_data = _generate_response(task_id, result)
    comm.send(response_data, dest=0, tag=22)


def main():
    comm: Intracomm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    size: int = comm.Get_size()

    if rank == MASTER_RANK:
        print(f"Running as master with size {size}")
        master_with_worker(comm, size)
    else:
        slave(comm, rank)


if __name__ == "__main__":
    main()
