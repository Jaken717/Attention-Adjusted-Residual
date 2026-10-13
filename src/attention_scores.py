#!/usr/bin/env python3
"""
Run one or many prompts through GPT-2 and dump both the generated
continuation and the per-layer self-attention scores.

For each prompt:
  1. Generate a continuation the normal way (sample.sample_sequence).
  2. Feed the full sequence (prompt + generated tokens) back through the
     model in a single forward pass to get self-attention weights for
     every layer/head over the whole sequence.
  3. Save the raw attention arrays (.npz), a per-layer heatmap image
     (averaged over heads), and print a short summary of which prompt
     tokens each generated token attended to most.

Usage:
    python src/attention_scores.py --prompt "The Eiffel Tower is not in Chicago. Therefore, the Eiffel Tower is in the city of"

    # many test cases at once, one prompt per line in a text file
    # (lines starting with # or blank lines are skipped)
    python src/attention_scores.py --prompts_file test_prompts.txt

Flags (same meaning as interactive_conditional_samples.py):
    --model_name, --models_dir, --length, --temperature, --top_k, --top_p, --seed
Extra flags:
    --out_dir           where results are written (default: attention_outputs)
    --top_attended      how many top source tokens to print per generated token (default 3)
    --keywords          comma-separated entity tokens to track specifically, e.g.
                         "Eiffel,Chicago,Paris,France,Berlin,Tower" (default). Each
                         occurrence in the full sequence (prompt + generated) is
                         located via BPE re-encoding and plotted individually, as two
                         separate pairs of figures: keyword_attention_by_layer.png /
                         keyword_attention_heads.png (post-softmax WEIGHTS, what the
                         model actually used) and keyword_attention_logits_by_layer.png /
                         keyword_attention_logits_heads.png (pre-softmax LOGITS, the raw
                         scaled Q.K score before competing with every other key).
    --query_pos         which token position's attention/candidate probabilities to
                         inspect (default: the last prompt token, i.e. the position
                         that determines the first generated word). Pass an int to
                         inspect a different position (e.g. a later generated token).
    --candidates        comma-separated answer words to compare head-to-head at the
                         query position, e.g. "Paris,Chicago" for the Eiffel Tower
                         negation prompts. For each candidate, computes the actual
                         model probability P(candidate | context up to query_pos)
                         via teacher forcing (handles multi-token words correctly),
                         prints a ranked table, and saves a bar chart. Off by default.
    --heatmap_head      which head to use for the full per-layer attention_heatmaps.png /
                         attention_logit_heatmaps.png grids (default 0). These show one
                         specific head's full seq x seq matrix per layer -- NOT averaged
                         over heads, since averaging smears out the sharp per-head
                         patterns -- styled after Figure 2 of Zhang et al. 2023,
                         "Efficient Streaming Language Models with Attention Sinks"
                         (arXiv:2309.17453): causally-masked future cells are shown in
                         flat gray, and each layer's panel is auto-scaled to its own
                         min/max rather than a single shared 0-1 scale. Both grids use
                         the same diverging RdBu_r palette (logits centered at zero,
                         weights centered at each panel's own mean).
    --true_answer       the real-world correct answer word (e.g. "Paris"), used only to
                         annotate attention_heatmaps.png / attention_logit_heatmaps.png:
                         every occurrence of this word in the sequence is marked green
                         (true token). Off by default (no ground truth is assumed).
    --hijack_word       the specific wrong-answer word to mark red (e.g. "Chicago"),
                         wherever it occurs in the sequence -- set this to compare the
                         SAME reference entity across multiple cases, even ones where
                         it wasn't the actual generated output. If omitted but
                         --true_answer is set, falls back to auto-detecting whatever
                         the model actually generated at the query position (if it
                         differs from true_answer).
"""

import os
import json
import math

import fire
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches

import model, sample, encoder


def slugify(text, max_len=40):
    keep = [c if c.isalnum() else '_' for c in text.strip()]
    slug = ''.join(keep).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug[:max_len] or 'prompt'


def load_prompts(prompt, prompts_file):
    prompts = []
    if prompts_file:
        with open(prompts_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                prompts.append(line)
    if prompt:
        prompts.append(prompt)
    if not prompts:
        raise ValueError('Provide --prompt "..." and/or --prompts_file path/to/file.txt')
    return prompts


def _heatmap_grid(matrices, tokens, title, out_path, center='zero',
                   true_positions=None, hijack_positions=None):
    """Grid of per-layer seq x seq heatmaps in the style of Zhang et al. 2023
    ("Efficient Streaming Language Models with Attention Sinks",
    arXiv:2309.17453, Figure 2): causally-invalid (future) cells are masked
    out in flat gray instead of being shown as misleading near-zero values,
    and each panel is scaled to its own range (via its own colorbar) rather
    than one shared scale, so layers with very different magnitude patterns
    are all individually legible.

    Always uses the same diverging RdBu_r palette (the one that made the
    logit heatmap easier to read than a flat sequential viridis scale):
      center='zero' -- vmin=-vmax, vcenter=0, vmax=vmax (for signed logits)
      center='mean' -- vmin=min, vcenter=mean, vmax=max (for non-negative
                        weights, which have no natural zero-centered
                        meaning, but still benefit from a diverging palette
                        that separates above-/below-average cells instead
                        of letting one saturated sink cell wash out everything
                        else on a 0..1 scale)

    true_positions / hijack_positions: optional sets of token indices to
    mark -- green for the true/correct-answer token, red for the
    hijacking token that the model actually (wrongly) generated. Shown as
    colored, bold tick labels when the sequence is short enough to label,
    else as thin ruled lines across every panel.
    """
    true_positions = set(true_positions or [])
    hijack_positions = set(hijack_positions or [])

    n_layer = len(matrices)
    cols = min(4, n_layer)
    rows = math.ceil(n_layer / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    label_tokens = len(tokens) <= 30

    cmap_obj = matplotlib.colormaps['RdBu_r'].copy()
    cmap_obj.set_bad('lightgray')

    for layer_idx, mat in enumerate(matrices):
        ax = axes[layer_idx]
        seq = mat.shape[0]
        future_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        masked = np.where(future_mask, np.nan, mat)
        valid = masked[~np.isnan(masked)]

        if not valid.size:
            vmin, vcenter, vmax = 0.0, 0.5, 1.0
        elif center == 'zero':
            vmax = float(np.abs(valid).max()) or 1e-6
            vmin, vcenter = -vmax, 0.0
        else:  # center == 'mean'
            vmin, vmax = float(valid.min()), float(valid.max())
            vcenter = float(valid.mean())
            if vmin == vmax:
                vmin, vmax = vmin - 1e-6, vmax + 1e-6
            if not (vmin < vcenter < vmax):
                vcenter = (vmin + vmax) / 2
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        im = ax.imshow(masked, cmap=cmap_obj, aspect='auto', norm=norm)
        ax.set_title(f'Layer {layer_idx}', fontsize=10)
        if label_tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=6)
            for i, lbl in enumerate(ax.get_xticklabels()):
                if i in hijack_positions:
                    lbl.set_color('red'); lbl.set_fontweight('bold')
                elif i in true_positions:
                    lbl.set_color('green'); lbl.set_fontweight('bold')
            for i, lbl in enumerate(ax.get_yticklabels()):
                if i in hijack_positions:
                    lbl.set_color('red'); lbl.set_fontweight('bold')
                elif i in true_positions:
                    lbl.set_color('green'); lbl.set_fontweight('bold')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            for pos in hijack_positions:
                ax.axvline(pos, color='red', linewidth=0.6, alpha=0.5)
                ax.axhline(pos, color='red', linewidth=0.6, alpha=0.5)
            for pos in true_positions:
                ax.axvline(pos, color='green', linewidth=0.6, alpha=0.5)
                ax.axhline(pos, color='green', linewidth=0.6, alpha=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n_layer:]:
        ax.axis('off')

    if true_positions or hijack_positions:
        handles = []
        if true_positions:
            handles.append(matplotlib.patches.Patch(color='green', label='true / correct-answer token'))
        if hijack_positions:
            handles.append(matplotlib.patches.Patch(color='red', label='hijacking / wrong-answer token'))
        fig.legend(handles=handles, loc='upper right', fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_layer_heatmaps(attn_weights, attn_logits, tokens, out_dir, head=0,
                         true_positions=None, hijack_positions=None):
    """Full per-layer self-attention heatmaps for ONE representative head
    (default head 0, matching the paper's convention) rather than averaged
    across heads -- averaging smears out the sharp, often near-one-hot
    patterns individual heads actually produce, which is why a naive
    head-averaged plot tends to look uninformative.

    Produces two grids, both using the same diverging RdBu_r palette:
      - attention_heatmaps.png        post-softmax WEIGHTS for this head
                                       (centered at each panel's own mean)
      - attention_logit_heatmaps.png  pre-softmax LOGITS for this head
                                       (centered at zero)
    """
    n_layer = len(attn_weights)
    weight_mats = [attn_weights[l][head] for l in range(n_layer)]
    logit_mats = [attn_logits[l][head] for l in range(n_layer)]

    _heatmap_grid(
        weight_mats, tokens,
        f'Self-attention WEIGHT per layer (head {head})  --  gray = causally masked',
        os.path.join(out_dir, 'attention_heatmaps.png'), center='mean',
        true_positions=true_positions, hijack_positions=hijack_positions)
    _heatmap_grid(
        logit_mats, tokens,
        f'Self-attention LOGIT per layer (head {head})  --  gray = causally masked',
        os.path.join(out_dir, 'attention_logit_heatmaps.png'), center='zero',
        true_positions=true_positions, hijack_positions=hijack_positions)


def find_keyword_spans(full_tokens, enc, keywords):
    """Locate every occurrence of each keyword in the token sequence.

    GPT-2's BPE splits most entity names into several sub-word tokens
    (e.g. "Eiffel" -> " E", "iff", "el"), so a keyword can't be matched
    against single decoded tokens. Instead we re-encode a handful of
    surface variants of each keyword and search for that exact id
    subsequence inside full_tokens.

    Returns a list of dicts: {keyword, start, end} (end is exclusive),
    one per match, in position order. Overlapping duplicate matches
    (same span found via two variants) are collapsed.
    """
    full_tokens = list(full_tokens)
    seen_spans = set()
    spans = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        variants = {kw, ' ' + kw, kw.lower(), ' ' + kw.lower(), kw.capitalize(), ' ' + kw.capitalize()}
        variant_ids = {tuple(enc.encode(v)) for v in variants if v}
        variant_ids.discard(())
        for ids in variant_ids:
            L = len(ids)
            for start in range(len(full_tokens) - L + 1):
                if tuple(full_tokens[start:start + L]) == ids:
                    span_key = (start, start + L)
                    if span_key not in seen_spans:
                        seen_spans.add(span_key)
                        spans.append({'keyword': kw, 'start': start, 'end': start + L})
    spans.sort(key=lambda s: s['start'])
    return spans


def _plot_line_by_layer(mat, labels, ylabel, title, out_path):
    """mat: [n_layer, n_head, n_spans]. Plots head-averaged value per layer,
    one line per keyword occurrence."""
    n_layer = mat.shape[0]
    avg_over_head = mat.mean(axis=1)  # [n_layer, n_spans]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s_idx, label in enumerate(labels):
        ax.plot(range(n_layer), avg_over_head[:, s_idx], marker='o', label=label)
    if avg_over_head.min() < 0 < avg_over_head.max():
        ax.axhline(0, color='grey', linewidth=0.7, zorder=0)
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(n_layer))
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _plot_heads_heatmap(mat, labels, title, out_path, diverging=False):
    """mat: [n_layer, n_head, n_spans]. Heatmap of every (layer, head) vs
    keyword occurrence."""
    n_layer, n_head, n_spans = mat.shape
    flat = mat.reshape(n_layer * n_head, n_spans)
    fig, ax = plt.subplots(figsize=(max(6, n_spans * 1.3), max(6, n_layer * n_head * 0.16)))
    if diverging:
        vmax = np.abs(flat).max() or 1.0
        im = ax.imshow(flat, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(flat, cmap='viridis', aspect='auto')
    ax.set_xticks(range(n_spans))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_layer * n_head))
    ax.set_yticklabels([f'L{l}H{h}' for l in range(n_layer) for h in range(n_head)], fontsize=5)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_keyword_attention(attn_weights, attn_logits, tokens, spans, query_pos, out_dir):
    """Plot attention paid specifically to keyword-token spans from a single
    query position (default: the last prompt token, i.e. the position whose
    output distribution determines the next generated word) — as two
    separate pairs of figures:

      - keyword_attention_by_layer.png / keyword_attention_heads.png:
        POST-SOFTMAX attention WEIGHTS (what the model actually used to mix
        values; always in [0, 1], competes/sums to 1 across all keys).
      - keyword_attention_logits_by_layer.png / keyword_attention_logits_heads.png:
        PRE-SOFTMAX attention LOGITS (the raw scaled Q.K score for this one
        key in isolation, unbounded and not normalized against other keys —
        shows "how much this head wanted to look here" before competition
        with every other token in the sequence).

    Returns (mat_weights, mat_logits, labels), each matrix shaped
    [n_layer, n_head, n_spans], or (None, None, None) if nothing to plot.
    """
    # Causal attention can't look forward, so only keep spans that end at or
    # before the query position.
    spans = [s for s in spans if s['end'] - 1 <= query_pos]
    if not spans:
        return None, None, None

    n_layer = len(attn_weights)
    n_head = attn_weights[0].shape[0]
    mat_w = np.zeros((n_layer, n_head, len(spans)))
    mat_l = np.zeros((n_layer, n_head, len(spans)))
    for l in range(n_layer):
        for s_idx, span in enumerate(spans):
            key_pos = span['end'] - 1
            mat_w[l, :, s_idx] = attn_weights[l][:, query_pos, key_pos]
            mat_l[l, :, s_idx] = attn_logits[l][:, query_pos, key_pos]

    labels = [f"{s['keyword']}@{s['end'] - 1}" for s in spans]
    query_label = f"{tokens[query_pos]!r} (pos {query_pos})"

    _plot_line_by_layer(
        mat_w, labels, 'Attention weight (avg over heads, post-softmax)',
        f'Attention WEIGHT from query {query_label} to keyword tokens',
        os.path.join(out_dir, 'keyword_attention_by_layer.png'))
    _plot_line_by_layer(
        mat_l, labels, 'Attention logit (avg over heads, pre-softmax Q.K/sqrt(d))',
        f'Attention LOGIT from query {query_label} to keyword tokens',
        os.path.join(out_dir, 'keyword_attention_logits_by_layer.png'))

    _plot_heads_heatmap(
        mat_w, labels, f'Attention WEIGHT to keyword tokens\nquery {query_label}',
        os.path.join(out_dir, 'keyword_attention_heads.png'))
    _plot_heads_heatmap(
        mat_l, labels, f'Attention LOGIT to keyword tokens\nquery {query_label}',
        os.path.join(out_dir, 'keyword_attention_logits_heads.png'), diverging=True)

    return mat_w, mat_l, labels


def summarize_keyword_attention(mat_w, mat_l, labels, top_n=5):
    """Text summary of which (layer, head) pairs attend most strongly to
    each keyword occurrence, by both weight and raw logit."""
    if mat_w is None:
        return '  (no keyword occurrences found before the query position)'
    _, n_head, _ = mat_w.shape
    lines = []
    for s_idx, label in enumerate(labels):
        col_w = mat_w[:, :, s_idx].reshape(-1)
        top_w = np.argsort(-col_w)[:top_n]
        parts_w = [f'L{i // n_head}H{i % n_head}={col_w[i]:.3f}' for i in top_w]
        lines.append(f"  {label:>20} <- top heads (weight): " + ', '.join(parts_w))
        col_l = mat_l[:, :, s_idx].reshape(-1)
        top_l = np.argsort(-col_l)[:top_n]
        parts_l = [f'L{i // n_head}H{i % n_head}={col_l[i]:.2f}' for i in top_l]
        lines.append(f"  {'':>20}    top heads (logit):  " + ', '.join(parts_l))
    return '\n'.join(lines)


def encode_candidate(enc, candidate):
    """Encode a candidate answer word/phrase the way it would actually occur
    as a continuation (leading space), e.g. 'Paris' -> ' Paris'."""
    candidate = candidate.strip()
    ids = enc.encode(' ' + candidate)
    if not ids:
        ids = enc.encode(candidate)
    return ids


def candidate_probabilities(sess, logits_tensor, full_seq_ph, prefix_ids, candidates, enc):
    """For each candidate answer (e.g. 'Paris', 'Chicago'), compute
    P(candidate | prefix) via teacher forcing: append the candidate's own
    token ids to prefix_ids, run one forward pass, and chain the per-token
    conditional probabilities. This handles multi-token candidates
    correctly (not just whichever single token happens to follow)."""
    prefix_ids = [int(t) for t in prefix_ids]
    results = []
    for cand in candidates:
        ids = encode_candidate(enc, cand)
        seq = prefix_ids + ids
        logits = sess.run(logits_tensor, feed_dict={full_seq_ph: [seq]})[0]  # [seq_len, vocab]
        start = len(prefix_ids) - 1
        logprob = 0.0
        step_probs = []
        for k, tok_id in enumerate(ids):
            row = logits[start + k]
            row = row - row.max()
            exp = np.exp(row)
            p = float(exp[tok_id] / exp.sum())
            step_probs.append(p)
            logprob += math.log(p + 1e-12)
        results.append({
            'candidate': cand,
            'token_ids': ids,
            'prob': math.exp(logprob),
            'logprob': logprob,
            'step_probs': step_probs,
        })
    results.sort(key=lambda r: -r['prob'])
    return results


def plot_candidate_probabilities(results, query_label, out_path):
    labels = [r['candidate'] for r in results]
    probs = [r['prob'] for r in results]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4.5))
    bars = ax.bar(labels, probs, color='#4C72B0')
    ax.set_ylabel('P(candidate | context)')
    ax.set_title(f'Answer-candidate probability given query {query_label}', fontsize=11)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{p:.3f}',
                 ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def summarize_candidate_probabilities(results):
    lines = []
    for r in results:
        lines.append(f"  {r['candidate']:>12}: p={r['prob']:.4f}  (logprob={r['logprob']:.3f}, tokens={r['token_ids']})")
    return '\n'.join(lines)


def plot_output_layer_heatmap(full_logits, tokens, candidates, enc, out_path, n_prompt_tokens=None):
    """Heatmap of the OUTPUT (final unembedding) layer: softmax(h @ wte^T)
    after ln_f, i.e. model.py's 'logits' -- not an attention layer at all.
    Unlike candidate_probabilities (one query position), this evaluates
    every position in the sequence in a single forward pass and plots
    P(candidate's first token) as a function of position, so you can see
    exactly where in the prompt the model's belief about each candidate
    answer shifts -- e.g. the moment "Chicago" overtakes "Paris" as you
    read through a negation sentence token by token.

    full_logits: [seq_len, n_vocab] raw logits (already np, no batch dim).
    """
    seq_len = full_logits.shape[0]
    cand_ids = [encode_candidate(enc, c)[0] for c in candidates]  # first token id only

    row_max = full_logits.max(axis=-1, keepdims=True)
    exp = np.exp(full_logits - row_max)
    sm = exp / exp.sum(axis=-1, keepdims=True)  # [seq_len, n_vocab]
    probs = sm[:, cand_ids].T  # [n_candidates, seq_len]

    from matplotlib.colors import LogNorm
    positive = probs[probs > 0]
    vmin = max(positive.min(), 1e-5) if positive.size else 1e-5
    vmax = max(probs.max(), vmin * 10)

    fig_w = max(8, seq_len * 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, 1.2 + 0.5 * len(candidates)))
    im = ax.imshow(probs, aspect='auto', cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels(candidates)
    if seq_len <= 60:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    if n_prompt_tokens is not None and 0 < n_prompt_tokens < seq_len:
        ax.axvline(n_prompt_tokens - 0.5, color='red', linewidth=1.2, linestyle='--')
        ax.text(n_prompt_tokens - 0.5, -0.7, 'prompt ends', color='red', fontsize=7,
                 ha='center', va='top', clip_on=False)
    ax.set_title('Output-layer probability of each candidate at every sequence position\n'
                  '(post ln_f + unembedding softmax -- NOT an attention layer)', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, label='P(candidate first token)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def summarize_attention(attentions, tokens, n_prompt_tokens, top_attended):
    """Print, for each generated token, which earlier tokens it attended to
    most (averaged across all layers and heads)."""
    stacked = np.stack(attentions, axis=0)  # [n_layer, n_head, seq, seq]
    combined = stacked.mean(axis=(0, 1))    # [seq, seq], avg over layer+head

    lines = []
    for pos in range(n_prompt_tokens, len(tokens)):
        row = combined[pos, :pos + 1]
        top_idx = np.argsort(-row)[:top_attended]
        parts = [f"{tokens[i]!r} ({row[i]:.3f})" for i in top_idx]
        lines.append(f"  generated {tokens[pos]!r:>12} <- attended most to: " + ', '.join(parts))
    return '\n'.join(lines)


def run(
    prompt=None,
    prompts_file=None,
    model_name='124M',
    models_dir='models',
    seed=None,
    length=20,
    temperature=1,
    top_k=40,
    top_p=1,
    out_dir='attention_outputs',
    top_attended=3,
    keywords='Eiffel,Chicago,Paris,France,Berlin,Tower',
    query_pos=None,
    candidates=None,
    heatmap_head=0,
    true_answer=None,
    hijack_word=None,
):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    prompts = load_prompts(prompt, prompts_file)
    if isinstance(keywords, str):
        keywords = keywords.split(',')
    keyword_list = [k.strip() for k in keywords if k.strip()] if keywords else []
    if isinstance(candidates, str):
        candidates = candidates.split(',')
    candidate_list = [c.strip() for c in candidates if c.strip()] if candidates else []

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

        full_seq = tf.placeholder(tf.int32, [1, None])
        analysis = model.model(hparams=hparams, X=full_seq, reuse=True)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        os.makedirs(out_dir, exist_ok=True)

        for i, prompt_text in enumerate(prompts):
            context_tokens = enc.encode(prompt_text)
            full_tokens = sess.run(output, feed_dict={context: [context_tokens]})[0]

            attn_arrays, attn_logit_arrays, full_logits = sess.run(
                [analysis['attentions'], analysis['attention_logits'], analysis['logits']],
                feed_dict={full_seq: [full_tokens]})
            attn_arrays = [a[0] for a in attn_arrays]              # drop batch dim -> [n_head, seq, seq]
            attn_logit_arrays = [a[0] for a in attn_logit_arrays]  # same, pre-softmax
            full_logits = full_logits[0]                           # output layer -> [seq, n_vocab]

            token_strs = [enc.decode([t]) for t in full_tokens]
            generated_text = enc.decode(full_tokens[len(context_tokens):])
            this_query_pos = query_pos if query_pos is not None else len(context_tokens) - 1

            case_dir = os.path.join(out_dir, f'{i:02d}_{slugify(prompt_text)}')
            os.makedirs(case_dir, exist_ok=True)

            with open(os.path.join(case_dir, 'generated.txt'), 'w', encoding='utf-8') as f:
                f.write('PROMPT:\n' + prompt_text + '\n\n')
                f.write('GENERATED CONTINUATION:\n' + generated_text + '\n\n')
                f.write('FULL TEXT:\n' + prompt_text + generated_text + '\n')

            np.savez(
                os.path.join(case_dir, 'attentions.npz'),
                **{f'layer_{l}': attn_arrays[l] for l in range(len(attn_arrays))},
            )
            with open(os.path.join(case_dir, 'tokens.json'), 'w', encoding='utf-8') as f:
                json.dump(token_strs, f, ensure_ascii=False, indent=2)

            true_positions, hijack_positions = set(), set()
            hijacked_word = None
            if true_answer:
                true_positions = {s['end'] - 1 for s in find_keyword_spans(full_tokens, enc, [true_answer])}
            if hijack_word:
                # explicit override: track this word regardless of what was actually
                # generated, so the same reference entity (e.g. "Chicago") can be
                # compared consistently across multiple cases.
                hijacked_word = hijack_word
                hijack_positions = {s['end'] - 1 for s in find_keyword_spans(full_tokens, enc, [hijacked_word])}
            elif true_answer:
                # auto-detect: whatever the model actually generated, if it differs
                # from the true answer.
                next_word = token_strs[this_query_pos + 1].strip() if this_query_pos + 1 < len(full_tokens) else None
                if next_word and next_word.lower() != true_answer.strip().lower():
                    hijacked_word = next_word
                    hijack_positions = {s['end'] - 1 for s in find_keyword_spans(full_tokens, enc, [hijacked_word])}

            plot_layer_heatmaps(attn_arrays, attn_logit_arrays, token_strs, case_dir, head=heatmap_head,
                                 true_positions=true_positions, hijack_positions=hijack_positions)

            print('=' * 80)
            print(f'[{i}] PROMPT: {prompt_text}')
            print(f'    GENERATED: {generated_text}')
            print(summarize_attention(attn_arrays, token_strs, len(context_tokens), top_attended))
            if true_answer:
                print(f'    true answer: {true_answer!r}  (found at positions {sorted(true_positions)})')
            if hijacked_word:
                print(f'    hijacking token: {hijacked_word!r}  (found at positions {sorted(hijack_positions)})')

            if keyword_list:
                spans = find_keyword_spans(full_tokens, enc, keyword_list)
                mat_w, mat_l, labels = plot_keyword_attention(
                    attn_arrays, attn_logit_arrays, token_strs, spans, this_query_pos, case_dir)
                if mat_w is not None:
                    np.savez(os.path.join(case_dir, 'keyword_attention.npz'),
                             weights=mat_w, logits=mat_l, labels=np.array(labels))
                print(f'    keyword attention (query={token_strs[this_query_pos]!r} @ pos {this_query_pos}):')
                print(summarize_keyword_attention(mat_w, mat_l, labels))

            if candidate_list:
                prefix_ids = full_tokens[:this_query_pos + 1]
                cand_results = candidate_probabilities(
                    sess, analysis['logits'], full_seq, prefix_ids, candidate_list, enc)
                query_label = f"{token_strs[this_query_pos]!r} (pos {this_query_pos})"
                plot_candidate_probabilities(
                    cand_results, query_label, os.path.join(case_dir, 'candidate_probabilities.png'))
                with open(os.path.join(case_dir, 'candidate_probabilities.json'), 'w', encoding='utf-8') as f:
                    json.dump(cand_results, f, indent=2)
                print(f'    candidate probabilities (query={query_label}):')
                print(summarize_candidate_probabilities(cand_results))

                plot_output_layer_heatmap(
                    full_logits, token_strs, candidate_list, enc,
                    os.path.join(case_dir, 'output_layer_heatmap.png'),
                    n_prompt_tokens=len(context_tokens))

            print(f'    saved -> {case_dir}/')

        print('=' * 80)


if __name__ == '__main__':
    fire.Fire(run)
