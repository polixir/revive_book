# 1. 制作REVIVE数据集

```python
cd revive_train/data
python process_data_to_revive.py
```

在revive_train/data目录下会生成Simglucose-revive.npz数据集文件

# 2. 训练虚拟环境

```python
cd revive_train
CUDA_VISIBLE_DEVICES=0 python train.py -df data/Simglucose-revive.npz -cf data/Simglucose_no_action_ts.yaml -rcf data/config.json -vm once -pm None --global_seed 42 --run_id benchmark --venv_algo revive_p --venv_rollout_horizon 100 --bc_loss nll --policy_hidden_layers 4 --policy_backbone res --rollout_plt_frequency 500
```

# 3. 训练策略

```python
cd revive_train
CUDA_VISIBLE_DEVICES=0 python train.py -df data/Simglucose-revive.npz -cf data/Simglucose_no_action_policy_ts_obs.yaml -rcf data/config.json -rf data/simglucose_reward.py -vm None -pm once --global_seed 42 --target_policy_name action --run_id benchmark --venv_algo revive_p --bc_loss nll --policy_hidden_layers 4 --policy_backbone res --bc_weight_decay 1e-7 --rollout_plt_frequency 500
```

# 4. 测试策略

```python
cd revive_train
python test_policy_ts_obs.py
```
