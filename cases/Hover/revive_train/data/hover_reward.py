import torch
from typing import Dict


def get_reward(data : Dict[str, torch.Tensor]):
    return torch.where((torch.abs(data["next_state"][...,0:1])<0.2) & (torch.abs(data["next_state"][...,1:2] - 1) < 0.2), 10,-0.3)