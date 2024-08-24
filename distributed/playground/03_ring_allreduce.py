# 0. set up a process group; 
# 1. implement ring "relay" all_reduce;  


# 0.0 set up boilerplate: p.start, rank/world_size? 
import os
import torch.distributed as dist
import torch
import torch.multiprocessing as mp

def fn():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank: {rank}, PID: {os.getpid()}] Running")

def ring_simple_relay_reduce():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank: {rank}, PID: {os.getpid()}] Running")
    
    # 0. data per process
    data = torch.ones(1) * rank

    # 1. set up communication neighbors
    prev = (rank - 1) % world_size
    next = (rank + 1) % world_size

    # 2. communication:
    # ------------------- Scatter -------------------
    # 2.1 recv from prev
    recv_buffers = torch.zeros(1)
    if rank != 0:
        dist.recv(tensor=recv_buffers, src=prev)
    
    # 2.2 reduce
    data = data + recv_buffers
    print(f"[Rank: {rank}, PID: {os.getpid()}] After Scatter, Result: {data}")

    # 2.3 send to next
    if rank != world_size - 1:
        dist.send(tensor=data, dst=next)
    
    # ------------------- Gather -------------------
    # 2.5 send the final result to the next
    if rank != world_size - 2:
        dist.send(tensor=data, dst=next)

    # 2.4 recv from prev 
    if rank != world_size - 1:
        dist.recv(tensor=data, src=prev)
    
    print(f"[Rank: {rank}, PID: {os.getpid()}] After Gather, Result: {data}")

def ring_all_reduce_sequential():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank: {rank}, PID: {os.getpid()}] Running")

    # 0. data per process
    # data = torch.arange(world_size) + rank * 10
    data = [torch.ones(1) * i + 1 for i in range(world_size)]
    # data = list(data.split(world_size))
    print(f"[Rank: {rank}, PID: {os.getpid()}] Original Data: {data}")

    # 1. set up communication neighbors
    prev_device = (rank - 1) % world_size
    next_device = (rank + 1) % world_size

    # 2. communication:
    # ------------------- Scatter -------------------
    for ii in range(world_size - 1):
        recv_idx = (rank - ii - 1) % world_size # decrement with ii 
        sent_idx = (rank - ii) % world_size # decrement with ii

        recv_buffers = torch.zeros(1)
        # Note: Without interleaving send/recv in different ranks, you would get deadlock.
        if rank % 2 == 0:
            dist.recv(tensor=recv_buffers, src=prev_device)
            dist.send(tensor=data[sent_idx], dst=next_device)
        else: 
            dist.send(tensor=data[sent_idx], dst=next_device)
            dist.recv(tensor=recv_buffers, src=prev_device)
        # print(f"[Rank: {rank}, PID: {os.getpid()}] Recv Data: {recv_buffers}")
        data[recv_idx] = data[recv_idx] + recv_buffers
        # print(f"[Rank: {rank}, PID: {os.getpid()}, Loop {ii}] After Scatter, Result: {data}")
    
    print(f"[Rank: {rank}, PID: {os.getpid()}] After Scatter, Result: {data}")
    
    # ------------------- Gather -------------------
    ii = 0 # rank p holds the final result of rank p + 1; need to pass that around;
    for ii in range(world_size - 1):
        send_idx = (rank - ii + 1) % world_size
        recv_idx = (rank - ii) % world_size
    
        if rank % 2 == 0:
            dist.send(tensor=data[send_idx], dst=next_device)
            dist.recv(tensor=data[recv_idx], src=prev_device)
        else:
            dist.recv(tensor=data[recv_idx], src=prev_device)
            dist.send(tensor=data[send_idx], dst=next_device)

    print(f"[Rank: {rank}, PID: {os.getpid()}] After Gather, Result: {data}")
    

def init_process(rank, world_size, fn, backend="gloo"):
    # 0.0 set up boilerplate: p.start, rank/world_size? 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    print(f"[Rank: {rank}, PID: {os.getpid()}] Starting")
    fn()
    # print(f"[Rank: {rank}, PID: {os.getpid()}] Done")

if __name__ == "__main__":
    fn = ring_all_reduce_sequential
    
    world_size = 4
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, fn))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()