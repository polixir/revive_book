import numpy as np

from revive.utils.common_utils import load_data


if __name__ == "__main__":
    data = load_data("Simglucose-high-100000-train.npz")

    for k, v in data.items():
        print(k, v.shape)
    
    # generate index info
    index = (np.where(np.sum((data['next_obs'][:-1] - data['obs'][1:]),axis=-1)!=0)[0]+1).tolist()+[data['obs'].shape[0]]
    index = [0]+index
    start = index[:-1]
    end = index[1:]
    traj = np.array(end) - np.array(start)

    # generate dataset for REVIVE
    outdata = {}
    outdata['obs'] = data['obs'][:, :1]
    outdata['next_obs'] = data['next_obs'][:, :1]

    outdata['property'] = data['obs'][:, 1:]
    outdata['delta_obs'] = outdata['next_obs'] - outdata['obs']
    
    outdata['action'] = data['action']
    outdata['index'] = (np.where(np.sum((outdata['next_obs'][:-1] - outdata['obs'][1:]),axis=-1)!=0)[0]+1).tolist() \
                        + [outdata['obs'].shape[0]]
    
    outdata_file = np.savez_compressed("Simglucose-revive.npz", **outdata)
    print("Done!")
