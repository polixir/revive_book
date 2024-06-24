import torch
import math
from typing import Dict


def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
        action = data['action'][...,0:1]
        u = torch.clamp(action, -2, 2)

        state = data['observation'][...,0:3]
        costheta = state[:,0].view(-1,1)
        sintheta = state[:, 1].view(-1,1)
        thdot = state[:, 2].view(-1,1)

        x = torch.acos(costheta)
        theta = ((x + math.pi) % (2 * math.pi)) - math.pi
        costs = theta ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        
        return -costs