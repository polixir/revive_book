CUDA_VISIBLE_DEVICES=1 \
python train.py \
-df data/Simglucose-revive.npz \
-cf data/Simglucose_no_action_policy_ts_obs.yaml \
-rcf data/config.json \
\
-rf data/simglucose_reward.py \
\
-vm None \
-pm once \
\
--global_seed 42 \
--target_policy_name action \
--run_id benchmark \
--venv_algo revive_p \
\
--bc_loss nll \
--policy_hidden_layers 4 \
--policy_backbone res \
--bc_weight_decay 1e-7 \
--rollout_plt_frequency 500 \

