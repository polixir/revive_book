import torch
import numpy as np 


def get_next_obs(data):
    obs = data["obs"]
    delta_obs = data["delta_obs"]

    if len(obs.shape) == 1:
        obs = obs.reshape(1, -1)
        delta_obs = delta_obs.reshape(1, -1)


    next_obs = obs + delta_obs

    if len(data["obs"].shape) == 1:
        next_obs = next_obs.reshape(-1)

    return next_obs


def get_truncat_ts_obs(data):
    ts_obs = data["ts_obs"]

    if len(ts_obs.shape) == 1:
        ts_obs = ts_obs.reshape(1, -1)

    truncat_ts_obs = ts_obs[..., :-5]

    if len(data["ts_obs"].shape) == 1:
        truncat_ts_obs = truncat_ts_obs.reshape(-1)

    return truncat_ts_obs


def get_truncat_ts_action(data):
    ts_action = data["ts_action"]

    if len(ts_action.shape) == 1:
        ts_action = ts_action.reshape(1, -1)

    truncat_ts_action = ts_action[..., :-5]

    if len(data["ts_action"].shape) == 1:
        truncat_ts_action = truncat_ts_action.reshape(-1)

    return truncat_ts_action
