python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_env.yaml -rcf ./data/bc.json -vm once -pm None --run_id bc_na
python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery.yaml -rf ./data/rocketrecovery_reward.py -rcf ./data/sac_config.json -vm None -pm once --run_id bc_sac


python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_env.yaml -rcf ./data/bc.json -vm once -pm None --run_id bc_na
python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery.yaml -rcf ./data/sac.json -rf ./data/rocketrecovery_reward.py  -vm None -pm once --run_id bc_na

python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_env.yaml -rcf ./data/ft.json -vm once -pm None --run_id bc_na_ft
python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery.yaml -rcf ./data/sac.json -rf ./data/rocketrecovery_reward.py  -vm None -pm once --run_id bc_na_ft


python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_ts_env.yaml -rcf ./data/bc.json -vm once -pm None --run_id bc_na_ts
python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_ts.yaml -rcf ./data/sac.json -rf ./data/rocketrecovery_reward.py  -vm None -pm once --run_id bc_na_ts

python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_ts_env.yaml -rcf ./data/ft.json -vm once -pm None --run_id bc_na_ft_ts

python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_ts.yaml -rcf ./data/sac.json -rf ./data/rocketrecovery_reward.py  -vm None -pm once --run_id bc_na_ft_ts



python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery_env.yaml -rcf ./data/bc.json -vm once -pm None --run_id bc_na


python train.py -df ./data/rocketrecovery.npz -cf ./data/rocketrecovery.yaml -rcf ./data/sac.json -rf ./data/rocketrecovery_reward.py -vm None -pm once --run_id bc_na