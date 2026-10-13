# 2026-10-13: Attention-adjusted residual (sum over heads)

## Idea

Route the skip connection of the attention sub-layer through the attention
weights, instead of adding the raw residual stream.

Original block (X is `[seq, 768]`, `X~ = ln_1(X)`, `h` indexes the 12 heads):

    X = X + sum_h A_h · X~ · W_v^h · W_o^h
    X = X + MLP(ln_2(X))

Modified block:

    X = sum_h A_h · (X + X~ · W_v^h · W_o^h)
      = (sum_h A_h) · X + a
    X = X + MLP(ln_2(X))            (unchanged)

where `A_h` is the post-softmax attention weight matrix of head `h` and `a` is
the normal attention output. Only the skip term changes: raw `X` is replaced
by `X` mixed across positions by the summed attention weights.

## Implementation

Branch `attention-experiments`, on top of checkpoint `736cd20`.

- `src/model.py`, `block()`:

      x = tf.matmul(tf.reduce_sum(w, axis=1), x) + a     # was: x = x + a

  `w` is `[batch, heads, dst, src]` (same tensor as `results['attentions']`).
- `src/model.py`, `default_hparams()`: new switch `attn_residual`, default
  `True`. `False` restores the original block.
- `src/sample.py`: when `attn_residual` is on, each generation step re-runs
  the full sequence instead of using the k/v `past` cache, because the new
  skip term needs `X` at every source position in every layer. `block()`
  asserts `past is None` in this mode.

## Setup

- Model: pretrained 124M checkpoint, no fine-tuning.
- Prompt: "The capital of France is"
- Decoding: 15 new tokens, `top_k=1` (greedy), `temperature=1`.
- Environment: TensorFlow 2.21 (CPU) in a temporary virtualenv; the machine
  had no TensorFlow installed.

## Results

| `attn_residual` | Continuation |
|---|---|
| `True` (new)  | `thelessthelessthelesstheless...` (one token repeated 15 times) |
| `False` (original) | ` the capital of the French Republic, and the capital of the French Republic is` |

- Logits are finite in both modes.
- Max absolute logit difference between the two modes on the prompt: 135.5.
- Sanity check: in both modes the sampler output equals an independent
  full-sequence greedy recompute, token for token.
- `attention_scores.py` was not run with the new residual.

## Observations

- With pretrained weights the new residual produces degenerate output.
- `sum_h A_h` has rows summing to 12 (each `A_h` row sums to 1), so the skip
  term scales the stream by about 12x per layer. Position 0 can only attend
  to itself, so it is multiplied by exactly 12 in each of the 12 layers. The
  attention output `a` and MLP output `m` keep their usual scale, so later
  layers contribute little relative to the skip term.
- Separately from scale, the skip path no longer preserves each position's
  own vector; it becomes a blend of earlier positions. The weights were not
  trained for this.
- Not yet determined which of the two effects (12x scale or position mixing)
  is responsible for the collapse.

## Next steps (open)

- Mean over heads (`tf.reduce_mean` instead of `tf.reduce_sum`) to remove the
  12x growth and isolate the effect of position mixing.
- Run `attention_scores.py` under the new residual and compare attention
  weights/logits with the baseline outputs.
- Fine-tuning with the new residual, if the pretrained weights cannot be
  used as is.
