import torch
import numpy as np


def get_reward(data):    
    obs = data["rocket_state"]
    action = data["engine_power"]
    next_obs = data["next_rocket_state"]
    singel_sample = False
    if len(obs.shape) == 1:
        obs = obs.reshape(1,-1)
        singel_sample = True
    if len(action.shape) == 1:
        action = action.reshape(1,-1)
    if len(next_obs.shape) == 1:
        next_obs = next_obs.reshape(1,-1)
    
    
    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch
        
    def get_shaping(state):
        shaping = (
            -100 * array_type.sqrt(state[...,0:1] * state[...,0:1] + state[...,1:2] * state[...,1:2])
            - 100 * array_type.sqrt(state[...,2:3] * state[...,2:3] + state[...,3:4] * state[...,3:4])
            - 100 * abs(state[...,4:5])
        )
        return shaping
    
    # 计算t时刻的位置信息
    prev_shaping = get_shaping(obs)
    # 计算t+1时刻的位置信息
    shaping = get_shaping(next_obs)
    
    # 当着陆器距离着陆平台越近/远时，奖励增加/减少
    reward = shaping - prev_shaping
    
    # 计算主引擎消耗
    m_power = (array_type.clip(action[...,0:1], 0.0, 1.0) + 1.0) * 0.5 
    m_power = array_type.where(action[...,0:1]>0.0, m_power, 0)
    # 计算左右引擎消耗
    s_power = array_type.clip(array_type.abs(action[...,1:2]), 0.5, 1.0)
    s_power = array_type.where(array_type.abs(action[...,1:2]) > 0.5,s_power, 0)
    
    # 副发动机喷射一次，奖励减少0.03分；主发动机喷射一次，奖励减少0.3分
    reward -= m_power * 0.3
    reward -= s_power * 0.03
    
    # 根据y轴坐标判断是否降落
    condition_1 = array_type.abs(next_obs[..., 1:2]) <= 0.05
    # 根据速度和角度信息判断状态是否安全
    condition_2 = (array_type.abs(next_obs[..., 4:5]) < 0.3) & (array_type.abs(next_obs[..., 2:3]) < 0.5) & (array_type.abs(next_obs[..., 3:4]) < 2)
    
    # 根据是否安全着陆而额外获得-100或+100分的奖励
    landing_reward = array_type.where(condition_2, 100, -100)
    reward = array_type.where(condition_1, landing_reward, reward)
    
    
    if singel_sample:
        reward = reward[0]
        if array_type == np:
            reward = reward.item()
            
    return reward