CUDA_VISIBLE_DEVICES=0 \
python train.py \
-df data/Simglucose-revive.npz \
-cf data/Simglucose_no_action_ts.yaml \
-rcf data/config.json \
\
-vm once \
-pm None \
\
--global_seed 42 \
--run_id benchmark \
--venv_algo revive_p \
\
--venv_rollout_horizon 100 \
\
--bc_loss nll \
--policy_hidden_layers 4 \
--policy_backbone res \
--rollout_plt_frequency 500 \
