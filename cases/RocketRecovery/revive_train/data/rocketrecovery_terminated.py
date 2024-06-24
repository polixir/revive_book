import torch
import numpy as np


def get_terminated(data):    
    next_rocket_state = data["next_rocket_state"]
    
    singel_sample = False
    if len(next_rocket_state.shape) == 1:
        next_rocket_state = next_rocket_state.reshape(1,-1)
        singel_sample = True
    
    if isinstance(next_rocket_state, np.ndarray):
        array_type = np
    else:
        array_type = torch
        
    # Status: landing and flight
    condition_1 = array_type.abs(next_rocket_state[..., 1:2]) <= 0.05
    
    done = array_type.where(condition_1, 1., 0.)
        
    if singel_sample:
        done = done[0]
        if array_type == np:
            done = done.item()

    return done