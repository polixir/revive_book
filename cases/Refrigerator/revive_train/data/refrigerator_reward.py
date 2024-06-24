import torch
from typing import Dict


def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    target_temperature = -2
    reward = - torch.abs(data['next_temperature'][...,0:1] - target_temperature)

    return reward