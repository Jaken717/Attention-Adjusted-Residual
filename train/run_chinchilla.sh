#!/bin/bash
# Chinchilla-optimal (2.5B tokens) from-scratch runs of baseline, approach_1, approach_2, one GPU each,
# plus the pretrained OpenAI GPT-2 124M evaluated on the same val split (GPU 3).
set -e
PY=/local/scratch/a/liu3170/.envs/gpt2train/bin/python
DATA=/local/scratch/a/liu3170/data/fineweb_edu
RUNS=/local/scratch/a/liu3170/runs/chinchilla
cd "$(dirname "$0")"
mkdir -p $RUNS
i=0
for v in baseline approach_1 approach_2; do
    CUDA_VISIBLE_DEVICES=$i $PY train.py --variant $v --data $DATA --out $RUNS/$v --tokens 2.5e9 > $RUNS/$v.log 2>&1 &
    i=$((i+1))
done
CUDA_VISIBLE_DEVICES=3 $PY eval_pretrained.py --data $DATA --out $RUNS/openai_gpt2_124M.json > $RUNS/openai_gpt2_124M.log 2>&1
wait
