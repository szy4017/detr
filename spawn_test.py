import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def fn(rank, ws, nums):
    dist.init_process_group('nccl', init_method='tcp://10.15.198.46:3311',
                            rank=rank, world_size=ws)
    rank = dist.get_rank()
    print(f"rank = {rank} is initialized")
    torch.cuda.set_device(rank)
    tensor = torch.tensor(nums).cuda()
    print(tensor)

if __name__ == "__main__":
    ws = 4
    mp.spawn(fn, nprocs=ws, args=(ws, [1, 2, 3, 4]))