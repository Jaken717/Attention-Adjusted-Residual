#!/bin/bash
# Build the Chinchilla attention-residual report once the runs are done.
set -e
PY=/local/scratch/a/liu3170/.envs/gpt2train/bin/python
DATA=/local/scratch/a/liu3170/data/fineweb_edu
RUNS=/local/scratch/a/liu3170/runs/chinchilla
OUT=/local/scratch/a/liu3170/gpt-2/experiment_log/2026-10-15_chinchilla_attention_residual
cd "$(dirname "$0")"
mkdir -p $OUT/data $OUT/figures
CUDA_VISIBLE_DEVICES=${GPU:-1} $PY analyze_collapse.py --data $DATA --runs $RUNS --out $OUT/data
$PY make_figures.py --runs $RUNS --probe $OUT/data --out $OUT/figures
for v in baseline approach_1 approach_2; do
    mkdir -p $OUT/data/$v
    cp $RUNS/$v/log.jsonl $RUNS/$v/config.json $OUT/data/$v/
    grep -v Warning $RUNS/$v.log > $OUT/data/$v/stdout.log
done
cp $RUNS/openai_gpt2_124M.json $OUT/data/
$PY report_numbers.py --runs $RUNS --probe $OUT/data --out $OUT > /dev/null
rm -f $OUT/data/attention_heatmaps.npz.tmp
cd $OUT
pdflatex -interaction=nonstopmode -halt-on-error report.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error report.tex | grep -E "Output written|^!"
rm -f report.aux report.log report.out
