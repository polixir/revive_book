import torch
import numpy as np


def next_Q_out_history(data):    
    Q_out_history = data["Q_out_history"]
    Q_out = data["Q_out"]
    
    singel_sample = False
    if len(Q_out_history.shape) == 1:
        Q_out_history = Q_out_history.reshape(1,-1)
        singel_sample = True

    if len(Q_out.shape) == 1:
        Q_out = Q_out.reshape(1,-1)
    
    if isinstance(Q_out, np.ndarray):
        array_type = np
    else:
        array_type = torch
        
    
    if isinstance(Q_out, np.ndarray):
        next_Q_out_history = np.concatenate([Q_out,Q_out_history[...,:-1]],-1)
    else:
        next_Q_out_history = torch.cat([Q_out,Q_out_history[...,:-1]],-1)

    if singel_sample:
        next_Q_out_history = next_Q_out_history[0]
        if array_type == np:
            next_Q_out_history = next_Q_out_history.item()

    return next_Q_out_history


def next_CV_in_history(data):    
    CV_in_history = data["CV_in_history"]
    CV_in = data["CV_in"]
    
    singel_sample = False
    if len(CV_in_history.shape) == 1:
        CV_in_history = CV_in_history.reshape(1,-1)
        singel_sample = True
    if len(CV_in.shape) == 1:
        CV_in = CV_in.reshape(1,-1)
    
    if isinstance(CV_in_history, np.ndarray):
        array_type = np
    else:
        array_type = torch
        
    
    if isinstance(CV_in, np.ndarray):
        next_CV_in_history = np.concatenate([CV_in,CV_in_history[...,:-1]],-1)
    else:
        next_CV_in_history = torch.cat([CV_in,CV_in_history[...,:-1]],-1)

    if singel_sample:
        next_CV_in_history = next_CV_in_history[0]
        if array_type == np:
            next_CV_in_history = next_CV_in_history.item()

    return next_CV_in_history