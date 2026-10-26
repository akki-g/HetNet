#!/bin/bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PYTHONPATH="$root/src:$root/src/envs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1

python src/main.py --experiment_name hetnet_a2c_fc_5x5_2p1a_reprod_shared_eval_seed_0 --save_dir /path/to/experiment/log --env_name fire_commander --nfriendly_P 2 --nfriendly_A 1 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 5 --lrate 0.001 --dim 5 --vision 1 --nfires 1 --batch_size 500 --max_steps 300 --hetgat --hetgat_a2c --shared_reward --seed 0  --eval --eval_string /path/to/save/eval/result  --eval_config 5x5_2p1a_initial_conditions.pkl --load /path/to/saved/model.pt

python src/print_plot_eval.py /path/to/save/eval/result
