import pickle
import numpy as np
import gymnasium
import neorl2


if __name__ == "__main__":
    policy_path = f"logs/simglucose-revive-no-action_horizon_100_ts/policy_train/policy.pkl"

    # venv_revive = pickle.load(open(venv_path, 'rb'))
    policy_revive = pickle.load(open(policy_path, 'rb'))
    env = gymnasium.make("Simglucose", mode='train')

    re_list = []
    for traj in range(100):
        obs, _ = env.reset()
        # venv_revive.reset()
        policy_revive.reset()
        re_turn = []
        done, trunc = False, False
        while not done and not trunc:
            action = policy_revive.infer({'obs': obs[:1], 'property': obs[1:]})
            # action = venv_revive.node_infer(node_name='action', state={'obs': obs})
            next_state, reward, done, trunc, info = env.step(action)
            re_turn.append(reward)
            obs = next_state
        
        print(len(re_turn), np.sum(np.array(re_turn)[:]))
        re_list.append(np.sum(re_turn))

    print('mean return:',np.mean(re_list), ' std:',np.std(re_list))
