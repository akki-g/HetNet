#!/bin/bash
# usage: run.sh <tag> <instrument 0|1> <real|binary>
SP=$(cd "$(dirname "$0")" && pwd); TAG=$1; INS=$2; V=$3
EXTRA=(); [ "$V" = binary ] && EXTRA=(--use_binary --msg_dim 16)
rm -rf "$SP/out_$TAG"; mkdir -p "$SP/out_$TAG"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 INSTRUMENT=$INS INSTRUMENT_OUT="$SP/out_$TAG/instrument.json"
/Users/akshatguduru/Desktop/Thesis/HetNet/.venv/bin/python "$SP/instrument.py" --env_name predator_capture --nfriendly_P 2 --nfriendly_A 1 --nagents 3 --hetgat --hetgat_a2c --num_epochs 1 --nprocesses 1 --epoch_size 4 --batch_size 500 --max_steps 80 --detach_gap 5 --lrate 0.001 --hid_size 128 --dim 5 --vision 2 --seed 991 --save_every 100 --experiment_name "p_$TAG" --save_dir "$SP/out_$TAG/ckpt" --metrics_file "$SP/out_$TAG/metrics.jsonl" --publication_env_version corrected-v1 --max_env_steps 0 --model_spec supplement-v1 --wall_seconds 0 --episode_log stdout --source_manifest "$SP/manifest.json" "${EXTRA[@]}" > "$SP/out_$TAG/stdout.log" 2> "$SP/out_$TAG/stderr.log"
echo "$TAG exit $?"
