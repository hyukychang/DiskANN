import time
from mpi4py import MPI
from mpi4py.MPI import Intracomm
import socket
import os
import struct
from typing import List, Tuple

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
MAP_SUFFIX = "_ids_uint32.bin"

QUERY_FILE = f"{DATA_PATH}/sift_query.fbin"
GT_FILE = f"{DATA_PATH}/sift_query_base_gt_100"

RESULT_PREFIX = f"{SAMPLE_PATH}/result/mem-"


UINT_8_SIZE = 1
UINT_32_SIZE = 4
INT_SIZE = 4
FLOAT_SIZE = 4
LONG_SIZE = 8

SEARCH_K = 10

L_LIST: str = "10 20 30 40 50 100 1000 10000"
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
    map_file = f"{INDEX_PREFIX}{data}{MAP_SUFFIX}"
    index_path_prefix = f"{INDEX_PATH}/{index_file}"
    map_path_prefix = f"{INDEX_PATH}/{map_file}"
    result_path = f"{RESULT_PREFIX}{data}"
    search_memory_index = f"{APP_PATH}/search_memory_index_with_id_map"
    command = (
        f"{search_memory_index} "
        "--data_type float "
        "--dist_fn l2 "
        f"--index_path_prefix {index_path_prefix} "
        f"--query_file {QUERY_FILE} "
        "-K 10 "
        f"-L {L_LIST} "
        f"--result_path {result_path} "
        f"--gt_file {GT_FILE} "
        f"--id_map_file {map_path_prefix}"
    )
    return command


def execute_command(task_id: int, data: int) -> int:
    host_id = socket.gethostname()
    print(f"{host_id} process for task_id : {task_id}")
    command = build_command(data)
    print(command)
    os.system(f"{command} > mem_search_{task_id}.txt")
    return data**2


def generate_tasks(size: int) -> List:
    return [{"task_id": i + 1, "data": i, "target_rank": i} for i in range(size)]


def _get_task_id(task: dict) -> int:
    return task["task_id"]


def _get_target_rank(task: dict) -> int:
    return task["target_rank"]


def _get_task_data(task: dict) -> int:
    return task.get("data", 0)


def _get_result_from_response(response_data: dict) -> int:
    return response_data["result"]


def _read_ground_truth_file(file_path: str) -> List:
    """
    return a list of ground truth vectors from the file
    """
    file_size: int = os.path.getsize(file_path)
    reader = open(file_path, mode="rb")
    num_of_points = int.from_bytes(reader.read(INT_SIZE), byteorder="little")
    dimension = int.from_bytes(reader.read(INT_SIZE), byteorder="little")
    print(f"reading {file_path}")
    print(f"Number of points: {num_of_points}, Dimension: {dimension}")
    ids = []
    for i in range(num_of_points):
        id = []
        for j in range(dimension):
            id.append(int.from_bytes(reader.read(UINT_32_SIZE), byteorder="little"))
        ids.append(id)
    does_file_have_distances: bool = (
        file_size == 2 * (num_of_points * dimension * UINT_32_SIZE) + 2 * UINT_32_SIZE
    )
    distances = []
    if does_file_have_distances:
        for i in range(num_of_points):
            dist = []
            for j in range(dimension):
                dist.append(struct.unpack("<f", reader.read(FLOAT_SIZE))[0])
            distances.append(dist)
    return ids, distances


def _read_result_ids_or_dist(file_path) -> List:
    """
    return a id list of search result path
    """
    print(f"reading {file_path}")
    reader = open(file_path, mode="rb")
    num_of_points = int.from_bytes(reader.read(INT_SIZE), byteorder="little")
    dimension = int.from_bytes(reader.read(INT_SIZE), byteorder="little")
    print(f"Number of points: {num_of_points}, Dimension: {dimension}")
    ids = []
    for i in range(num_of_points):
        id = []
        for j in range(dimension):
            id.append(int.from_bytes(reader.read(UINT_32_SIZE), byteorder="little"))
        ids.append(id)
    return ids


def format_result_file(result_prefix: str) -> List[Tuple[int, float]]:
    # def format_result_file(result_prefix: str) -> list[tuple[int, float]]:
    id_path = f"{result_prefix}_idx_uint32.bin"
    dist_path = f"{result_prefix}_dists_float.bin"
    ids = _read_result_ids_or_dist(id_path)
    distances = _read_result_ids_or_dist(dist_path)
    query_num = len(ids)
    result = []
    for i in range(query_num):
        temp = []
        for j in range(len(ids[i])):
            temp.append((ids[i][j], distances[i][j]))
        result.append(temp)
    return result


def _merge_result(result: List, K: int) -> list:
    """
    merge the search result of each L
    result = [
        [quey_0_result, query_1_result, ...], : query list of node 0 // for L
        [quey_0_result, query_1_result, ...], : query list of node 1
        ...
    ]
    """
    # print(result[0][0])
    print(len(result))
    query_len = len(result[0][0])
    merged_list = []

    for L_IDX in range(len(L_LIST.split())):
        L_result = []
        for query_idx in range(query_len):
            merged_result: set = set()
            for node_result in result:
                for data in node_result[L_IDX][query_idx]:
                    merged_result.add(data)
            merged_result = list(merged_result)
            merged_result = sorted(merged_result, key=lambda x: x[1])[:K]
            L_result.append(merged_result)
        merged_list.append(L_result)

    return merged_list


def _calculate_recall(query_result: List, ground_truth_ids: List, K: int) -> float:
    """
    calculate recall of the search result
    """
    L_list = L_LIST.split()
    merged_count = [0] * len(L_list)
    query_num = len(ground_truth_ids)
    for idx_L in range(len(L_list)):  # iterate over L
        for query_id in range(query_num):
            L = L_list[idx_L]
            gt_id = ground_truth_ids[query_id][:SEARCH_K]

            search_result = query_result[idx_L][query_id]
            # print(len(search_result))
            for id, dist in search_result:
                # print(id)
                if id in gt_id:
                    merged_count[idx_L] += 1

    print(f"{'L':>10}{f'merged_search_reacall@{SEARCH_K}':>30}")
    print("=" * 70)
    for idx_L in range(len(L_list)):
        merged_recall = f"{merged_count[idx_L] / (SEARCH_K * query_num) * 100:.2f}"
        print(f"{L_list[idx_L]:>10}{merged_recall:>30}")


def master_with_worker(comm: Intracomm, size: int):
    # tasks = range(1, size)  # Number of tasks equal to the number of slaves
    results = []
    tasks = generate_tasks(size=size)

    send_start_time = time.time()
    for task in tasks:
        destination = _get_target_rank(task)
        if destination == MASTER_RANK:
            # don't have to send task to self
            print("do not have to send")
            continue
        else:
            comm.send(task, dest=destination, tag=MASTER_TO_WORKER)
            print(f"Master sent data {task} to slave {task['task_id']}")

    query_results = []
    for task in tasks:
        destination = _get_target_rank(task)
        if destination == MASTER_RANK:
            result = execute_command(
                task_id=_get_task_id(task), data=_get_task_data(task)
            )
            results.append(result)
            print("appending query result...")
            query_result = []
            for L in L_LIST.split():
                result_prefix = f"{RESULT_PREFIX}{_get_task_data(task)}_{L}"
                data = format_result_file(result_prefix)
                query_result.append(data)
            query_results.append(query_result)
        else:
            response_data = comm.recv(source=destination, tag=WORKER_TO_MASTER)
            result = _get_result_from_response(response_data)
            results.append(result)
            query_results.append(response_data["data"])
            print(f"Master received result {result} from slave {task}")
    receive_end_time = time.time()


    gt_ids, gt_dist = _read_ground_truth_file(GT_FILE)
    merged_result = _merge_result(query_results, SEARCH_K)
    print(
        "The time of execution of above program is :",
        (receive_end_time - send_start_time) * 10**3,
        "ms",
    )
    recall = _calculate_recall(merged_result, gt_ids, SEARCH_K)
    print(f"Recall: {recall}")
    print("All tasks completed. Results:", results)


def _generate_response(task_id: int, result: int, data: List) -> dict:
    return {"task_id": task_id, "result": result, "data": data}


def slave(comm: Intracomm, rank: int):
    task = comm.recv(source=MASTER_RANK, tag=MASTER_TO_WORKER)
    task_id = _get_task_id(task)
    data = _get_task_data(task)
    print(data)
    result = execute_command(task_id, data)
    query_results = []
    for L in L_LIST.split():
        result_prefix = f"{RESULT_PREFIX}{data}_{L}"
        query_result = format_result_file(result_prefix)
        query_results.append(query_result)
    print(f"worker {rank} finished task {task_id}")
    response_data = _generate_response(task_id, result=result, data=query_results)
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
