import pickle
import numpy as np
import gymnasium
import neorl2


if __name__ == "__main__":
    env = gymnasium.make("Simglucose", mode='train')

    re_list = []
    for traj in range(100):
        obs, _ = env.reset()
        re_turn = []
        done, trunc = False, False
        while not done and not trunc:
            action = env.action_space.sample()
            # action = venv_revive.node_infer(node_name='action', state={'obs': obs})
            next_state, reward, done, trunc, info = env.step(action)
            re_turn.append(reward)
            obs = next_state
        
        print(len(re_turn), np.sum(np.array(re_turn)[:]))
        re_list.append(np.sum(re_turn))

    print('mean return:',np.mean(re_list), ' std:',np.std(re_list))
