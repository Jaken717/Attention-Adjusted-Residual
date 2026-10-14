# 2026-10-14: Head-wise attention residual, A_h (V_h + X_h)

Replaces the method from
[2026-10-13](2026-10-13_attention_adjusted_residual.md).

## Idea

Feed the raw residual stream into attention as an extra value term, per head,
and drop the separate skip connection around attention.

Original block (X is `[seq, 768]`, `X~ = ln_1(X)`, `h` indexes the 12 heads):

    Y = Concat_h[A_h V_h] W_O + b_O,   V_h = X~ W_v^h
    X = X + Y
    X = X + MLP(ln_2(X))

Modified block:

    Y = Concat_h[A_h (V_h + X_h)] W_O + b_O
    X = Y
    X = X + MLP(ln_2(X))            (unchanged)

`X_h` is head `h`'s 64-dimensional slice of the raw (pre-`ln_1`) stream `X`,
split the same way as `V`.

Difference from the 2026-10-13 method, `X = (sum_h A_h) X + a`:

- each head mixes only its own 64-dim slice of `X`, not the full 768 dims;
- the mixed `X` goes through `W_O` instead of bypassing it;
- `A_h` rows sum to 1 and each head contributes one slice, so there is no
  12x growth from summing heads.

## Implementation

- `src/model.py`, `attn()`: new `residual_x` argument; when `attn_residual`
  is on, `x_heads = split_heads(residual_x)` and `multihead_attn()` computes
  `A_h (V_h + X_h)`. The k/v cache (`present`) still stores plain `V`.
- `src/model.py`, `block()`: passes `residual_x=x`; with `attn_residual`,
  `x = a` replaces `x = x + a`.
- `attn_residual` switch and full-sequence sampling in `src/sample.py`
  are unchanged from 2026-10-13.
- Ported from a rewrite of `model.py` (saved as `stash@{0}`) that used TF1
  imports and dropped the attention outputs. The port keeps
  `tensorflow.compat.v1`, the `attentions` / `attention_logits` outputs and
  the shape comments. Logits from the port and from the rewrite (with only
  its imports switched to TF2) are identical: max abs difference 0.0.

## Setup

Same as 2026-10-13: pretrained 124M checkpoint, prompt
"The capital of France is", 15 greedy tokens (`top_k=1`), TensorFlow 2.21 CPU.

## Results

| `attn_residual` | Continuation |
|---|---|
| `True` (new) | ` PSU PSU PSU ...` (one token repeated 15 times) |
| `False` (original) | ` the capital of the French Republic, and the capital of the French Republic is` |

- Max absolute logit difference between the two modes on the prompt: 97.5
  (135.5 for the 2026-10-13 method).
- Sampler output equals a full-sequence greedy recompute in both modes.

## Observations

- With pretrained weights this also collapses to a repeated token, although
  the 12x scale growth of the previous method is gone.
- Likely contributors: each position's own `X` is no longer carried forward
  directly but is blended over earlier positions by `A_h`; and it now passes
  through `W_O`, which was trained to map attention outputs, not the
  residual stream. Not yet tested which matters more.

## Next steps (open)

- Fine-tune with the new block, since the pretrained weights were not
  trained for it.
- Run `attention_scores.py` under the new residual.
