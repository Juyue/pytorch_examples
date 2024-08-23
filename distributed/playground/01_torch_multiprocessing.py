# Goal: create several processes using torch.multiprocessing.

import torch.multiprocessing as mp
import time
import os

def run_simple_process(rank, world_size, shared_array):
    print(f"Hello, World! I am {rank} of {world_size}")
    print(f"my pid is {os.getpid()}")
    shared_array[0] = rank
    time.sleep(3)
    print(f"Goodbye, World! I am {rank} of {world_size}")

def main_simple_multi_process():
    print(f"Main process: Hello, World")

    world_size = 4
    processes = []
    shared_array = mp.Array("i", range(10))
    print(f"Main process: my pid is {os.getpid()}")
    print(f"Main process: shared_array is {shared_array[:]}")
    # import pdb; pdb.set_trace()

    for rank in range(world_size):
        p = mp.Process(target=run_simple_process, args=(rank, world_size, shared_array))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        print(f"Main process: Goodbye, World")
    
    print(f"Main process: shared_array is {shared_array[:]}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main_simple_multi_process()
    
    
    
    
    

