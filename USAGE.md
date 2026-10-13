# Running GPT-2 locally (test prompts)

This repo's original code requires TensorFlow 1.12, which does not run on
Apple Silicon (arm64). A dedicated conda environment was set up instead,
using tensorflow-macos, with the source files patched to run in TF1-compat
mode. See "Setup notes" at the bottom for what was changed.

## Environment

Conda env name: `gpt2`
Location: `/opt/miniconda3/envs/gpt2`

Activate it either by:
```
conda activate gpt2
```
or call its python binary directly without activating:
```
/opt/miniconda3/envs/gpt2/bin/python ...
```

Run all commands below from the repo root: `/Users/jinkailiu/Desktop/gpt-2`

## One-time: download a model checkpoint

Already downloaded: `124M` (in `models/124M/`).

To get other sizes:
```
/opt/miniconda3/envs/gpt2/bin/python download_model.py 355M
/opt/miniconda3/envs/gpt2/bin/python download_model.py 774M
/opt/miniconda3/envs/gpt2/bin/python download_model.py 1558M
```
Bigger models = slower generation and more RAM, but better quality.

## Test prompts (interactive)

```
export PYTHONIOENCODING=UTF-8
/opt/miniconda3/envs/gpt2/bin/python src/interactive_conditional_samples.py --top_k 40
```

You'll be prompted:
```
Model prompt >>>
```
Type any text and press Enter — it prints one or more generated completions.
Press Ctrl+C to quit.

### Useful flags

```
/opt/miniconda3/envs/gpt2/bin/python src/interactive_conditional_samples.py \
  --model_name=124M \
  --length=40 \
  --temperature=0.7 \
  --top_k=40 \
  --nsamples=3 \
  --batch_size=1
```

- `--model_name`  which downloaded model to use (`124M`, `355M`, `774M`, `1558M`)
- `--length`      number of tokens to generate (default: half the model's context window)
- `--temperature` randomness; lower = more deterministic, higher = more random (default 1)
- `--top_k`       0 = no restriction; 40 is a good default for coherent output
- `--nsamples`    how many completions to generate per prompt
- `--batch_size`  must evenly divide nsamples; only affects speed/memory

See all flags:
```
/opt/miniconda3/envs/gpt2/bin/python src/interactive_conditional_samples.py -- --help
```

## Unconditional generation (no prompt, just random samples)

```
/opt/miniconda3/envs/gpt2/bin/python src/generate_unconditional_samples.py --top_k 40 | tee /tmp/samples
```

Same flags apply (model_name, length, temperature, top_k, nsamples, batch_size).

## Attention scores (per-layer, per-head)

`src/attention_scores.py` runs a prompt, generates a continuation, and also
dumps the model's self-attention weights for every layer (averaged over
heads for the plot; raw per-head values in the saved `.npz`).

Single prompt:
```
/opt/miniconda3/envs/gpt2/bin/python src/attention_scores.py \
  --prompt "The Eiffel Tower is not in Chicago. Therefore, the Eiffel Tower is in the city of" \
  --length=20
```

Many test cases at once — edit `test_prompts.txt` (one prompt per line,
`#`-prefixed or blank lines ignored) and run:
```
/opt/miniconda3/envs/gpt2/bin/python src/attention_scores.py --prompts_file test_prompts.txt --length=20
```

`--prompt` and `--prompts_file` can be combined; all other flags from
`interactive_conditional_samples.py` apply (`--model_name`, `--temperature`,
`--top_k`, `--top_p`, `--seed`), plus:
- `--out_dir`       where results go (default `attention_outputs/`)
- `--top_attended`  how many top source tokens to print per generated token (default 3)

For every prompt this creates `attention_outputs/<NN>_<slug>/` containing:
- `generated.txt` — prompt, generated continuation, and full text
- `attentions.npz` — one array per layer, shape `[n_head, seq_len, seq_len]`
- `tokens.json` — the decoded token for each position (for labeling the arrays)
- `attention_heatmaps.png` — one heatmap per layer, attention averaged over heads

It also prints, for each generated token, the top attended-to source tokens
(averaged across all layers/heads) — e.g. does it attend to "not" or
"Chicago" when producing its answer.

Note: attention is computed by feeding the full sequence (prompt + generated
tokens) through the model once with teacher forcing, after generation — so
it reflects genuine self-attention, not an approximation.

## Note on training

This repo (openai/gpt-2) only ships inference/sampling code — there is no
training script. Fine-tuning on your own text is not supported here; that
requires a different codebase (e.g. a community fork like nshepperd/gpt-2).

## Setup notes (what was changed, for reference)

- Created conda env `gpt2`, native arm64, Python 3.10.
- Installed `tensorflow-macos` (2.16) instead of the repo's pinned TF 1.12,
  since TF 1.12 has no arm64 wheel and won't run under Rosetta (missing AVX
  support in emulation).
- Patched these files to use `tensorflow.compat.v1` in TF1-graph mode:
  - `src/model.py` — also replaced the removed `tensorflow.contrib.training.HParams`
    with a small local `HParams` class.
  - `src/sample.py`
  - `src/interactive_conditional_samples.py`
  - `src/generate_unconditional_samples.py`
- Installed `fire`, `regex`, `tqdm`, `requests` (from `requirements.txt`) and
  `matplotlib` (for attention heatmaps) into the `gpt2` env.
- `src/model.py`'s `attn`/`block`/`model` functions now also return/collect
  the raw attention weights per layer (`results['attentions']`), used by
  `src/attention_scores.py`. This is additive — `sample.py` and the other
  scripts are unaffected since they only read `results['logits']`/`['present']`.


/opt/miniconda3/envs/gpt2/bin/python src/attention_scores.py --prompts_file test_prompts.txt --length=20
