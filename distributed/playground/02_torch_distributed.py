import torch.multiprocessing as mp
import os
import torch.distributed as dist
import torch

def run_dummy(rank, world_size):
    print(f"[PID {os.getpid()} Rank: {rank}], Running function")

def run_point_communication(rank, world_size):
    # do point to point communication
    print(f"[PID {os.getpid()} Rank: {rank}], Running function")

    tensor = torch.zeros(1)
    if rank == 0: 
        tensor += 1
        # send tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print(f"[PID {os.getpid()} Rank: {rank}], Sending Tensor: {tensor}")
    elif rank == 1:
        dist.recv(tensor=tensor, src=0)
        print(f"[PID {os.getpid()} Rank: {rank}], Receiving Tensor: {tensor}")
    else:
        print(f"[PID {os.getpid()} Rank: {rank}], Rank {rank} is not involved in point to point communication")

def run_all_reduce(rank, world_size):
    print(f"[PID {os.getpid()} Rank: {rank}], Running function")

    tensor = torch.ones(1)
    group = dist.new_group([0, 1])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"[PID {os.getpid()} Rank: {rank}], Tensor: {tensor}")

def run_all_gather(rank, world_size):
    print(f"[PID {os.getpid()} Rank: {rank}], Running function")

    tensor = torch.ones(1)
    group_size = 2
    group = dist.new_group(range(group_size))
    gathered_list = [torch.zeros(1) for _ in range(group_size)]
    dist.all_gather(gathered_list, tensor, group=group)
    print(f"[PID {os.getpid()} Rank: {rank}], Tensor: {tensor}")
    print(f"[PID {os.getpid()} Rank: {rank}], Tensor List: {gathered_list}")

def init_process(rank, world_size, fn, backend="gloo"):

    # for each process, assign it to a process_group?
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"[PID {os.getpid()} Rank: {rank}], Hello, World!")
    fn(rank, world_size)
    print(f"[PID {os.getpid()} Rank: {rank}], Done")

if __name__ == "__main__":
    fn = run_dummy
    fn = run_point_communication
    fn = run_all_reduce
    fn = run_all_gather
    world_size = 4
    processes = []
    # Question: fork/spawn? Code example to show the difference?
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, fn))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()