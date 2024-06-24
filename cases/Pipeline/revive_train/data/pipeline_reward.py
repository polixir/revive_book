import torch
import numpy as np


def get_reward(data):    
    next_Q_out = data["next_Q_out"]
    Q_target = data["Q_target"]

    singel_sample = False
    
    if len(Q_target.shape) == 1:
        Q_target = Q_target.reshape(1,-1)
        singel_sample = True
    if len(next_Q_out.shape) == 1:
        next_Q_out = next_Q_out.reshape(1,-1)
    
    
    if isinstance(next_Q_out, np.ndarray):
        array_type = np
    else:
        array_type = torch
    
    reward = array_type.square((200 - array_type.abs(Q_target - next_Q_out)) * 0.01) - 3
    #reward = reward - (action*action) * 0.01
    
    if singel_sample:
        reward = reward[0]
        if array_type == np:
            reward = reward.item()

    return reward