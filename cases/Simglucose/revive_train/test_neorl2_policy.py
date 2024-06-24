import json
import pickle
import numpy as np
import torch

import gymnasium
import neorl2
from neorl2.utils.g2p2c_utils.policy_wrapper import PolicyWrapper


if __name__ == "__main__":
    config_path = "/home/ubuntu/chenjiawei/NeoRL2/scripts/logs/Simglucose/config.json"
    # Load the JSON file
    with open(config_path, "r") as json_file:
        config = json.load(json_file)

    model_path = "/home/ubuntu/chenjiawei/NeoRL2/scripts/logs/Simglucose/models/best_model.pth"
    policy = PolicyWrapper(config, model_path)

    env = gymnasium.make("Simglucose", mode='train')

    re_list = []
    for traj in range(100):
        obs, _ = env.reset()
        re_turn = []
        done, trunc = False, False
        while not done and not trunc:
            action = policy.predict(obs)
            # action = venv_revive.node_infer(node_name='action', state={'obs': obs})
            next_state, reward, done, trunc, info = env.step(action)
            re_turn.append(reward)
            obs = next_state
        
        print(len(re_turn), np.sum(np.array(re_turn)[:]))
        re_list.append(np.sum(re_turn))

    print('mean return:',np.mean(re_list), ' std:',np.std(re_list))
