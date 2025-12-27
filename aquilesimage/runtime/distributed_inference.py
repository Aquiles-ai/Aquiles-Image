import torch

def get_device_count():
    return torch.cuda.device_count()