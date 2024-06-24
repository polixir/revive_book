import pickle
import numpy as np
import neorl2


if __name__ == "__main__":
    policy_path = "logs/bc_na_2/policy.pkl"

    policy_revive = pickle.load(open(policy_path, 'rb'))
    env = neorl2.make('Pipeline')

    re_list = []
    for traj in range(100):
        obs,_ = env.reset()
        re_turn = []
        done = False
        timeout = False
        while not done and not timeout:
            action = policy_revive.infer({'Q_out': obs[...,:1],'Q_target': obs[...,1:2],'Q_out_history':obs[...,2:27],'CV_in_history':obs[...,27:]})
            next_state, reward, done, timeout,_ = env.step(action)
            re_turn.append(reward)
            obs = next_state
        
        print(len(re_turn), np.sum(np.array(re_turn)[:]))
        re_list.append(np.sum(re_turn))

    print('mean return:',np.mean(re_list), ' std:',np.std(re_list), ' normal_score:', env.get_normalized_score(np.mean(re_list)))
