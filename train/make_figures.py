"""Figures for the Chinchilla attention-residual training report."""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ['baseline', 'approach_1', 'approach_2']
LABEL = {'baseline': 'baseline (standard GPT-2)', 'approach_1': 'approach_1', 'approach_2': 'approach_2'}
COLOR = {'baseline': '#2a78d6', 'approach_1': '#eb6834', 'approach_2': '#1baf7a'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 2,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'figure.dpi': 150,
})


def load_log(path):
    val, train = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (val if 'val_loss' in r else train).append(r)
    return val, train


def ema(xs, alpha=0.05):
    out, m = [], None
    for x in xs:
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return np.array(out)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.pdf'))
    fig.savefig(os.path.join(out, name + '.png'))
    plt.close(fig)


def fig_val(logs, ref, uni, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for v in VARIANTS:
        val = [r for r in logs[v][0] if not r.get('final')]  # final eval uses the full 20M split; plot 4M points only
        x = [r['tokens'] / 1e9 for r in val]
        y = [r['val_loss'] for r in val]
        ax.plot(x, y, '-o', ms=4, color=COLOR[v], label=LABEL[v])
        if v != 'approach_2':
            nonfinal = [r for r in val if not r.get('final')]
            x, y = [r['tokens'] / 1e9 for r in nonfinal], [r['val_loss'] for r in nonfinal]
            txt = f'{y[-1]:.3f}' if v == 'baseline' else \
                f'{y[-1]:.3f} / {logs["approach_2"][0][-1]["val_loss"]:.3f}'
            ax.annotate(txt, (x[-1], y[-1]), xytext=(6, -10 if v == 'baseline' else 9), textcoords='offset points',
                        va='center', color=INK2, fontsize=8)
    ax.axhline(uni, color=MUTED, ls=':', lw=1.2)
    ax.text(0.98, uni - 0.1, f'unigram model (no context) = {uni:.3f}', color=INK2, fontsize=8, ha='right',
            va='top', transform=ax.get_yaxis_transform())
    ax.axhline(ref, color=MUTED, ls='--', lw=1.2)
    ax.text(0.02, ref - 0.12, f'OpenAI GPT-2 124M (pretrained) = {ref:.3f}', color=INK2, fontsize=8, va='top',
            transform=ax.get_yaxis_transform())
    ax.set(xlabel='training tokens (billions)', ylabel='validation loss (nats/token)', ylim=(2.8, 8.2),
           xlim=(-0.05, 2.75))
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.45))
    save(fig, out, 'fig_val_loss')


def fig_train(logs, uni, out):
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.0), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1.2]})
    for v in VARIANTS:
        tr = logs[v][1]
        s = np.array([r['step'] for r in tr])
        axes[0].plot(s, [r['train_loss'] for r in tr], color=COLOR[v], lw=0.6, alpha=0.25)
        axes[0].plot(s, ema([r['train_loss'] for r in tr]), color=COLOR[v], label=LABEL[v])
        axes[1].plot(s, ema([r['grad_norm'] for r in tr]), color=COLOR[v], label=LABEL[v])
        if v == 'baseline':
            axes[2].plot(s, [r['lr'] * 1e4 for r in tr], color=INK2)
    stop = {v: logs[v][1][-1]['step'] for v in VARIANTS[1:]}
    for v, st in stop.items():
        axes[0].annotate(f'{v} stopped (step {st})', (st, 7.68), xytext=(-4, 10 if v == 'approach_1' else -14),
                         textcoords='offset points', ha='right', fontsize=7.5, color=INK2,
                         arrowprops={'arrowstyle': '-', 'color': MUTED, 'lw': 0.8})
    axes[0].axhline(uni, color=MUTED, ls=':', lw=1.2)
    axes[0].set(ylabel='train loss', ylim=(2.8, 11.2), title='Training loss (EMA, raw in light)')
    axes[0].legend(loc='center right', bbox_to_anchor=(1.0, 0.35))
    axes[1].set(ylabel='grad norm (pre-clip)', yscale='log', title='Global gradient norm (EMA)')
    axes[2].set(ylabel='lr (×1e-4)', xlabel='step (524,288 tokens / step)', title='Learning rate')
    fig.tight_layout()
    save(fig, out, 'fig_train_dynamics')


def fig_layers(probe, out):
    models = probe['models']
    trained = {v: next(k for k in models if k.startswith(v + '@step')) for v in VARIANTS}
    panels = [('ent_ratio', 'attention entropy / uniform', False, (0, 1.05)),
              ('self_w', 'attention on own position A[i,i]', False, None),
              ('cos', 'cosine between positions', False, (0, 1.05)),
              ('rank1_res', 'distance from rank 1', False, (0, 1.05)),
              ('norm', 'residual-stream norm', True, None)]
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.0))
    for ax, (key, title, logy, ylim) in zip(axes, panels):
        for v in VARIANTS:
            for tag, ls, mk, alpha in (('init', '--', None, 0.8), ('trained', '-', 'o', 1.0)):
                k = f'{v}@init' if tag == 'init' else trained[v]
                rows = [r for r in models[k]['layers'] if key in r]
                xs = [-1 if r['layer'] == 'emb' else r['layer'] for r in rows]
                ax.plot(xs, [r[key] for r in rows], ls=ls, marker=mk, ms=3, color=COLOR[v], alpha=alpha,
                        lw=1.6 if tag == 'trained' else 1.2)
        ax.set(title=title, xlabel='layer', xticks=[-1, 0, 3, 6, 9, 11], xticklabels=['emb', 0, 3, 6, 9, 11])
        if logy:
            ax.set_yscale('log')
        if ylim:
            ax.set_ylim(*ylim)
    handles = [plt.Line2D([], [], color=COLOR[v], lw=2, label=LABEL[v]) for v in VARIANTS]
    handles += [plt.Line2D([], [], color=INK2, ls='--', lw=1.2, label='at initialization'),
                plt.Line2D([], [], color=INK2, marker='o', ms=3, lw=1.6, label='trained (last checkpoint)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, out, 'fig_layer_diagnostics')


def fig_heatmaps(heat_path, probe, out):
    h = np.load(heat_path)
    models = probe['models']
    trained = {v: next(k for k in models if k.startswith(v + '@step')) for v in VARIANTS}
    layers = [0, 5, 11]
    fig, axes = plt.subplots(len(VARIANTS), len(layers), figsize=(7.2, 7.4))
    for i, v in enumerate(VARIANTS):
        for j, l in enumerate(layers):
            a = h[f'{trained[v]}|L{l}'].mean(0)  # mean over heads
            ax = axes[i, j]
            im = ax.imshow(a, cmap='Blues', vmin=0, vmax=0.5, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(f'layer {l}')
            if j == 0:
                ax.set_ylabel(f'{v}\n(query position)', color=INK2)
            if i == len(VARIANTS) - 1:
                ax.set_xlabel('key position', color=INK2)
    fig.suptitle('Attention patterns, mean over 12 heads, first 48 tokens of a val sequence', fontsize=10)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label('attention weight (clipped at 0.5)', color=INK2)
    save(fig, out, 'fig_attention_heatmaps')


def fig_unigram(pred_path, probe, out):
    p = np.load(pred_path)
    models = probe['models']
    uni = p['unigram']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    rng = np.random.default_rng(0)
    sel = np.concatenate([np.argsort(-uni)[:2000], rng.choice(len(uni), 3000, replace=False)])
    for ax, v in zip(axes, VARIANTS):
        k = next(k for k in models if k.startswith(v + '@step'))
        ax.scatter(uni[sel], p[k][sel], s=4, color=COLOR[v], alpha=0.35, linewidths=0)
        ax.plot([1e-9, 1], [1e-9, 1], color=MUTED, ls='--', lw=1)
        ax.set(xscale='log', yscale='log', xlim=(1e-8, 0.1), ylim=(1e-8, 0.1),
               xlabel='unigram frequency of token', title=f'{v}: KL(model‖unigram) = {models[k]["kl_to_unigram"]:.3f}')
    axes[0].set_ylabel('model mean predicted probability')
    fig.tight_layout()
    save(fig, out, 'fig_unigram_match')


def fig_toy(out, T=1024, D=64, L=12, seed=0):
    """Rank collapse under stacked causal uniform averaging, with and without an identity skip."""
    rng = np.random.default_rng(seed)
    P = np.tril(np.ones((T, T))) / np.arange(1, T + 1)[:, None]  # uniform causal attention
    X0 = rng.standard_normal((T, D))

    def dist(X):
        Xs = X[SKIP:]
        return np.linalg.norm(Xs - Xs.mean(0)) / np.linalg.norm(Xs)

    curves = {}
    X = X0.copy(); c = [dist(X)]
    for _ in range(L):
        X = P @ X; c.append(dist(X))
    curves['mix residual: $X \\leftarrow PX$ (approach_1/2, attention uniform)'] = (c, COLOR['approach_1'])
    X = X0.copy(); c = [dist(X)]
    for _ in range(L):
        W = rng.standard_normal((D, D)) * 0.02 * np.sqrt(D) / np.sqrt(2 * L)
        X = X + P @ X @ W; c.append(dist(X))
    curves['identity skip: $X \\leftarrow X + PXW$ (baseline-style)'] = (c, COLOR['baseline'])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for lab, (c, col) in curves.items():
        ax.plot(range(L + 1), c, '-o', ms=4, color=col, label=lab)
    ax.set(xlabel='number of layers', ylabel='distance from rank 1', ylim=(0, 1.05),
           title='Toy model: random $X\\in\\mathbb{R}^{1024\\times64}$, causal uniform $P$')
    ax.legend(loc='center right')
    save(fig, out, 'fig_toy_rank_collapse')


SKIP = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logs = {v: load_log(os.path.join(args.runs, v, 'log.jsonl')) for v in VARIANTS}
    probe = json.load(open(os.path.join(args.probe, 'collapse_probe.json')))
    matched = json.load(open(os.path.join(args.probe, 'matched_evals.json')))['val4M']
    fig_val(logs, matched['openai_gpt2_124M'], matched['unigram'], args.out)  # same 4M val subset as the curves
    uni = probe['unigram_val_ce']
    fig_train(logs, uni, args.out)
    fig_layers(probe, args.out)
    fig_heatmaps(os.path.join(args.probe, 'attention_heatmaps.npz'), probe, args.out)
    fig_unigram(os.path.join(args.probe, 'mean_predictions.npz'), probe, args.out)
    fig_toy(args.out)


if __name__ == '__main__':
    main()
