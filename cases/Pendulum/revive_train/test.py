import pickle
import numpy as np
import gym


if __name__ == "__main__":
    policy_path = "logs/pendulum/policy.pkl"

    policy_revive = pickle.load(open(policy_path, 'rb'))
    env = gym.make('Pendulum')

    re_list = []
    for traj in range(100):
        obs = env.reset()
        re_turn = []
        done = False
        timeout = False
        while not done:
            action = policy_revive.infer({'observation': obs})
            next_state, reward, done,_ = env.step(action)
            re_turn.append(reward)
            obs = next_state
        
        print(len(re_turn), np.sum(np.array(re_turn)[:]))
        re_list.append(np.sum(re_turn))

    print('mean return:',np.mean(re_list), ' std:',np.std(re_list))
