import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def get_device_count():
    return torch.cuda.device_count()