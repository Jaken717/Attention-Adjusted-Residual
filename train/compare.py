"""Summarize and plot the Chinchilla runs: val/train loss curves and a final table."""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = ['baseline', 'approach_1', 'approach_2']
COLORS = {'baseline': '#2a6fdb', 'approach_1': '#e8590c', 'approach_2': '#2b8a3e'}


def load(path):
    val, train = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (val if 'val_loss' in r else train).append(r)
    return val, train


def smooth(xs, k=50):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = None
    ref_path = os.path.join(args.runs, 'openai_gpt2_124M.json')
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = []
    for v in VARIANTS:
        p = os.path.join(args.runs, v, 'log.jsonl')
        if not os.path.exists(p):
            continue
        val, train = load(p)
        axes[0].plot([r['tokens'] / 1e9 for r in val], [r['val_loss'] for r in val], 'o-', ms=3,
                     color=COLORS[v], label=v)
        axes[1].plot([r['step'] for r in train], smooth([r['train_loss'] for r in train]),
                     color=COLORS[v], label=v)
        final = next((r for r in val if r.get('final')), None)
        tps = [r['tok_per_s'] for r in train[20:]]
        rows.append({'variant': v,
                     'final_val_loss': final['val_loss'] if final else None,
                     'last_val_loss': val[-1]['val_loss'] if val else None,
                     'steps': train[-1]['step'] + 1 if train else 0,
                     'tok_per_s': sum(tps) / len(tps) if tps else None})
    if ref is not None:
        axes[0].axhline(ref, color='gray', ls='--', label='OpenAI GPT-2 124M (pretrained)')
        rows.append({'variant': 'openai_gpt2_124M_pretrained', 'final_val_loss': ref})
    axes[0].set(xlabel='training tokens (B)', ylabel='val loss (FineWeb-Edu)', title='Validation loss')
    axes[1].set(xlabel='step', ylabel='train loss (50-step mean)', title='Training loss')
    lo = min([r['final_val_loss'] or r.get('last_val_loss') or 9 for r in rows] + [9])
    for ax in axes:
        ax.set_ylim(lo - 0.2, lo + 2.0)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'loss_curves.png'), dpi=150)
    fig.savefig(os.path.join(args.out, 'loss_curves.pdf'))
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
