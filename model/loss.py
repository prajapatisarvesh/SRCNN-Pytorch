import torch
import torch.nn.functional as F

def byte_mse_loss(output, target):
    return (0.5*(output-target) ** 2).mean(dtype=torch.float64)