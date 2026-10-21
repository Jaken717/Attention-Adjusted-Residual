"""Figures and numbers.tex for the 2026-10-20 report: approach_1/2/4 trained with the annealed diagonal bias.

Reads the run logs and the outputs of annealed_approaches_analysis.py; writes figures/*.pdf (+ .png) and numbers.tex.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
UNIGRAM = 7.591  # unigram cross-entropy of this val split (2026-10-19 report)
# Categorical slots 1-6 of the validated reference palette, in fixed order. baseline, three_residual and
# approach_3 keep their colors from the earlier reports. Markers give a second, non-color channel.
C = {'baseline': '#2a78d6', 'three_residual': '#eb6834', 'approach_1': '#1baf7a', 'approach_3': '#eda100',
     'approach_2': '#e87ba4', 'approach_4': '#008300'}
MK = {'baseline': 'o', 'three_residual': 's', 'approach_1': 'D', 'approach_2': '^', 'approach_4': 'v', 'approach_3': 'P'}
LS = {'baseline': '-', 'three_residual': '-', 'approach_1': '-', 'approach_2': '-', 'approach_4': '-', 'approach_3': ':'}
LABEL = {'baseline': 'baseline (GPT-2) + bias', 'three_residual': 'three_residual + bias', 'approach_1': 'approach_1 + bias',
         'approach_2': 'approach_2 + bias', 'approach_4': 'approach_4 + bias', 'approach_3': 'approach_3 (no bias)'}
RUN = {'baseline': 'baseline_diag8_anneal1000', 'three_residual': 'tr_diag8_anneal1000',
       'approach_1': 'approach_1_diag8_anneal1000', 'approach_2': 'approach_2_diag8_anneal1000',
       'approach_4': 'approach_4_diag8_anneal1000', 'approach_3': 'approach_3'}
NEW = ['approach_1', 'approach_2', 'approach_4']

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 1.8,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'figure.dpi': 150,
})


def load_log(path):
    val, train = [], []
    for line in open(path):
        r = json.loads(line)
        (val if 'val_loss' in r else train).append(r)
    return val, train


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'))
    plt.close(fig)


def spread(ys, gap):
    order = np.argsort(ys); out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        out[b] = max(out[b], out[a] + gap)
    return out


def rolling_median(x, w):
    x = np.asarray(x, float)
    return np.array([np.median(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def heat(ax, a, title=None):
    im = ax.imshow(a, cmap='Blues', vmin=0, vmax=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(MUTED)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def curve(val):
    v = [r for r in val if not r.get('final')]
    return [r['step'] for r in v], [r['val_loss'] for r in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='/local/scratch/a/liu3170/runs/openwebtext')
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    D = os.path.join(args.dir, 'data'); F = os.path.join(args.dir, 'figures')
    os.makedirs(F, exist_ok=True)
    probe = json.load(open(os.path.join(D, 'probe_init.json')))
    tr = json.load(open(os.path.join(D, 'trained.json')))
    maps = np.load(os.path.join(D, 'attention.npz'))
    logs = {k: load_log(os.path.join(args.runs, v, 'log.jsonl')) for k, v in RUN.items()}
    nobias = {k: load_log(os.path.join(args.runs, f'{k}_nobias_s500', 'log.jsonl')) for k in NEW}
    layers = np.arange(-1, 12)
    nums = {}

    # 1. final loss (dot plot); approach_1 shows its best checkpoint loss and is marked as diverged
    fin = {k: next(r['val_loss'] for r in logs[k][0] if r.get('final')) for k in RUN if k != 'approach_1'}
    a1v = [r for r in logs['approach_1'][0] if not r.get('final')]
    a1best = min(a1v, key=lambda r: r['val_loss'])
    rows = [('approach_3', fin['approach_3'], 'final'), ('baseline', fin['baseline'], 'final'),
            ('three_residual', fin['three_residual'], 'final'), ('approach_2', fin['approach_2'], 'final'),
            ('approach_1', a1best['val_loss'], f"best, step {a1best['step']}; diverged later"),
            ('approach_4', fin['approach_4'], 'collapsed')]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 2.5), gridspec_kw={'width_ratios': [2.2, 1]}, sharey=True)
    for ax, lo, hi in [(axs[0], 3.3, 3.85), (axs[1], 7.2, 7.75)]:
        for y, (k, v, note) in enumerate(rows):
            if lo <= v <= hi:
                ax.plot([lo, v], [y, y], color=GRID, lw=1.2, zorder=1)
                ax.plot(v, y, MK[k], ms=8, color=C[k], zorder=2, mfc='white' if k == 'approach_1' else C[k], mew=1.8)
                ax.text(v + (hi - lo) * 0.02, y + 0.3, f'{v:.3f}' + ('' if note == 'final' else f'  {note}'),
                        fontsize=7.5, color=INK2)
        ax.set_xlim(lo, hi); ax.grid(axis='y', visible=False)
    axs[1].axvline(UNIGRAM, color=MUTED, ls=':', lw=1.2)
    axs[1].text(UNIGRAM - 0.01, 0.0, 'unigram', fontsize=7.5, color=INK2, ha='right')
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([LABEL[r[0]] for r in rows]); axs[0].set_ylim(-0.6, len(rows) - 0.3)
    axs[0].invert_yaxis()
    axs[0].set_xlabel('val loss (lower is better)'); axs[1].set_xlabel('val loss (broken axis)')
    axs[1].spines['left'].set_visible(False); axs[1].tick_params(left=False)
    save(fig, F, 'fig_final_loss')

    # 2. init probe: rank-1 distance per layer, without and with the bias
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.1), sharey=True)
    for ax, b, title in [(axs[0], '0', '(a) no bias'), (axs[1], '8', '(b) self-logit bias $b=8$')]:
        names = ['baseline', 'approach_1', 'approach_2', 'approach_4', 'three_residual']
        for k in names:
            ax.plot(layers, probe[f'{k}|b{b}']['rank1'], marker=MK[k], ms=3.2, color=C[k],
                    ls='--' if k == 'three_residual' else '-')
        labs = [('baseline', 'baseline'), ('approach_1', 'approach_1, approach_2' if b == '0' else 'approach_1'),
                ('approach_4', 'approach_4'), ('three_residual', 'three_residual')] + ([('approach_2', 'approach_2')] if b == '8' else [])
        ends = [probe[f'{k}|b{b}']['rank1'][-1] for k, _ in labs]
        for (k, lab), yl in zip(labs, spread(ends, 0.06)):
            ax.text(11.35, yl, lab, fontsize=7.5, color=INK2, va='center')
        ax.plot([-1.3, 11], [0.15, 0.15], color=INK2, lw=0.9, ls='--')
        if b == '8':
            ax.text(-1, 0.17, 'collapse line (~0.15)', fontsize=7.5, color=INK2)
        ax.set_title(title, fontsize=9.5, loc='left'); ax.set_xlabel('layer (−1 = embeddings)')
        ax.set_xlim(-1.3, 15.5); ax.set_ylim(0, 1.05); ax.set_xticks(range(-1, 12, 2))
    axs[0].set_ylabel('distance from rank 1 at init')
    save(fig, F, 'fig_init_probe')

    # 3. pretraining: bias schedule, val loss, zoom
    fig = plt.figure(figsize=(8.4, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 2.4, 2.1], hspace=0.4)
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    trn = logs['approach_2'][1]
    a0.plot([r['step'] for r in trn], [r['diag_bias'] for r in trn], color=INK2)
    a0.axvspan(0, 1000, color=GRID, alpha=0.6, lw=0)
    a0.text(560, 6.2, '$b = 8\\,(1 - t/1000)$', fontsize=8, color=INK2)
    a0.text(2900, 2.0, '$b = 0$: every model runs exactly as defined', ha='center', fontsize=8, color=INK2)
    a0.set_ylabel('bias $b$'); a0.set_ylim(-0.4, 8.6); a0.set_xlim(0, 4800); a0.set_yticks([0, 4, 8])
    a0.set_title('(a) self-logit bias schedule (all bias runs)', fontsize=9.5, loc='left')
    order = ['baseline', 'three_residual', 'approach_1', 'approach_2', 'approach_4']
    for ax, title in [(a1, '(b) validation loss (4M-token eval every 250 steps)'),
                      (a2, '(c) zoom on the healthy band')]:
        for k in order:
            x, y = curve(logs[k][0])
            ax.plot(x, y, ls=LS[k], marker=MK[k], ms=3.2, color=C[k], label=LABEL[k])
        ax.axvspan(0, 1000, color=GRID, alpha=0.6, lw=0)
        ax.set_title(title, fontsize=9.5, loc='left'); ax.set_ylabel('val loss')
    a1.axhline(UNIGRAM, color=MUTED, ls=':', lw=1.2)
    a1.text(4750, UNIGRAM + 0.15, f'unigram (no context) {UNIGRAM:.2f}', ha='right', fontsize=7.5, color=INK2)
    x, y = curve(logs['approach_4'][0])
    a1.annotate('approach_4 collapses\nonce the bias is gone', (1250, y[x.index(1250)]), xytext=(1500, 9.3),
                fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    x, y = curve(logs['approach_1'][0])
    a1.annotate('approach_1 diverges;\nstopped at step 3690', (x[-1], y[-1]), xytext=(3300, 5.6), fontsize=7.5,
                color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    a1.set_xlim(0, 4800); a1.set_ylim(3, 11.2); a1.legend(loc='upper right', ncol=2, fontsize=7.5)
    a2.set_xlim(950, 5600); a2.set_ylim(3.3, 4.85); a2.set_xlabel('optimizer step (524,288 tokens each)')
    ks = ['baseline', 'three_residual', 'approach_2']
    lasts = [curve(logs[k][0])[1][-1] for k in ks]
    for k, v, yl in zip(ks, lasts, spread(lasts, 0.07)):
        a2.text(4790, yl, f'{k} {v:.3f}', fontsize=7.5, color=INK2, va='center')
    save(fig, F, 'fig_training')

    # 4. gap to the baseline+bias control at matched steps
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    bx, by = curve(logs['baseline'][0]); bmap = dict(zip(bx, by))
    for k in ['three_residual', 'approach_1', 'approach_2']:
        x, y = curve(logs[k][0])
        pts = [(s, v - bmap[s]) for s, v in zip(x, y) if s in bmap and s >= 250]
        ax.plot(*zip(*pts), marker=MK[k], ms=3.2, color=C[k], label=LABEL[k])
    ax.axvspan(0, 1000, color=GRID, alpha=0.6, lw=0); ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xlim(200, 4800); ax.set_ylim(-0.05, 0.75); ax.set_xlabel('optimizer step')
    ax.set_ylabel('val loss − control'); ax.legend(loc='upper right', fontsize=7.5)
    save(fig, F, 'fig_gap')

    # 5. gradient norm (pre-clip), rolling median over 50 steps, log scale
    fig, axs = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True, gridspec_kw={'hspace': 0.15})
    for k in order:
        t = logs[k][1]
        axs[0].plot([r['step'] for r in t], rolling_median([r['grad_norm'] for r in t], 50), color=C[k], lw=1.5,
                    label=LABEL[k])
        axs[1].plot([r['step'] for r in t], rolling_median([r['train_loss'] for r in t], 50), color=C[k], lw=1.5)
    axs[0].set_yscale('log'); axs[0].axhline(1.0, color=INK2, lw=0.9, ls='--')
    axs[0].text(4780, 1.15, 'clip threshold 1.0', fontsize=7.5, color=INK2, ha='right')
    axs[0].set_ylabel('grad norm before clipping\n(50-step rolling median)')
    axs[0].legend(loc='upper left', ncol=3, fontsize=7.5, bbox_to_anchor=(0, 1.22))
    axs[1].set_ylabel('train loss\n(50-step rolling median)'); axs[1].set_ylim(3.2, 8)
    axs[1].set_xlabel('optimizer step')
    for ax in axs:
        ax.axvspan(0, 1000, color=GRID, alpha=0.6, lw=0); ax.set_xlim(0, 4800)
    save(fig, F, 'fig_grad_norm')

    # 6. zoom on the end of annealing: approach_2 vs approach_4
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for k in ['approach_2', 'approach_4']:
        t = [r for r in logs[k][1] if 500 <= r['step'] <= 1500]
        s = [r['step'] for r in t]
        axs[0].plot(s, rolling_median([r['train_loss'] for r in t], 20), color=C[k], label=LABEL[k])
        axs[1].plot(s, rolling_median([r['grad_norm'] for r in t], 20), color=C[k])
    axs[1].set_yscale('log')
    for ax, title in [(axs[0], '(a) train loss'), (axs[1], '(b) grad norm before clipping')]:
        ax.axvline(1000, color=INK2, lw=0.9, ls='--'); ax.set_title(title, fontsize=9.5, loc='left')
        ax.set_xlabel('step'); ax.set_xlim(500, 1500)
        for st, bl in [(500, 'b=4'), (750, 'b=2'), (1000, 'b=0')]:
            ax.text(st + 8, 0.02, bl, transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va='bottom')
    axs[0].legend(loc='upper left', fontsize=7.5, bbox_to_anchor=(0, 0.93))
    # (c) val loss with the bias as stored vs forced to 0, at the repro checkpoints
    ax = axs[2]
    for k in ['approach_2', 'approach_4']:
        pts = sorted((tr[f'{k} step {s}']['step'], tr[f'{k} step {s}']['val_loss_bias'], tr[f'{k} step {s}']['val_loss_nobias'])
                     for s in [500, 750, 1000, 1250] if f'{k} step {s}' in tr)
        if not pts:
            continue
        s, vb, v0 = zip(*pts)
        ax.plot(s, vb, '-', marker=MK[k], color=C[k], ms=4)
        ax.plot(s, v0, '--', marker=MK[k], color=C[k], ms=4, mfc='white')
    ax.plot([], [], '-', color=INK2, label='bias as scheduled'); ax.plot([], [], '--', color=INK2, mfc='white', label='bias forced to 0')
    ax.axhline(UNIGRAM, color=MUTED, ls=':', lw=1.2)
    ax.text(880, UNIGRAM + 0.08, 'unigram', fontsize=7, color=INK2, ha='center')
    ax.legend(loc='upper center', fontsize=7.5); ax.set_xlabel('checkpoint step'); ax.set_xticks([500, 750, 1000, 1250])
    ax.set_title('(c) val loss at checkpoints', fontsize=9.5, loc='left')
    save(fig, F, 'fig_anneal_end')

    # 7. no-bias controls on OpenWebText (first 500 steps)
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for k in NEW:
        x, y = curve(logs[k][0]); keep = [i for i, s in enumerate(x) if s <= 500]
        ax.plot([x[i] for i in keep], [y[i] for i in keep], marker=MK[k], ms=4, color=C[k], label=LABEL[k])
        x, y = curve(nobias[k][0])
        ax.plot(x, y, '--', marker=MK[k], ms=4, color=C[k], mfc='white', label=k + ', no bias')
    ax.axhline(UNIGRAM, color=MUTED, ls=':', lw=1.2)
    ax.text(510, UNIGRAM + 0.12, 'unigram', fontsize=7.5, color=INK2)
    ax.set_xlabel('optimizer step'); ax.set_ylabel('val loss'); ax.set_xticks([0, 250, 500]); ax.set_ylim(4.8, 11.2)
    ax.legend(loc='upper right', ncol=2, fontsize=7.5)
    save(fig, F, 'fig_nobias')

    # 8. self-weight and rank-1 distance of the trained models
    trained_keys = [('baseline final', 'baseline'), ('three_residual final', 'three_residual'),
                    ('approach_1 step 3000', 'approach_1'), ('approach_2 final', 'approach_2'),
                    ('approach_4 final', 'approach_4')]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for name, k in trained_keys:
        lab = LABEL[k].replace(' + bias', '') + (' (step 3000)' if k == 'approach_1' else '')
        axs[0].plot(range(12), tr[name]['self_w'], marker=MK[k], ms=3.5, color=C[k], label=lab)
        axs[1].plot(layers, tr[name]['rank1'], marker=MK[k], ms=3.5, color=C[k])
    axs[0].set_xlabel('layer'); axs[0].set_ylabel('mean attention on itself, $A_{ii}$'); axs[0].set_ylim(0, 1)
    axs[0].set_xticks(range(12)); axs[0].set_title('(a) self-weight (bias is 0 in all)', fontsize=9.5, loc='left')
    axs[0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.2), ncol=5, fontsize=7.5)
    axs[1].axhline(0.15, color=INK2, lw=0.9, ls='--'); axs[1].set_ylim(0, 1.02); axs[1].set_xticks(range(-1, 12, 2))
    axs[1].set_xlabel('layer (−1 = embeddings)'); axs[1].set_ylabel('distance from rank 1')
    axs[1].set_title('(b) rank-1 distance of the residual stream', fontsize=9.5, loc='left')
    save(fig, F, 'fig_trained_diag')

    # 9. attention maps of the trained models
    fig, axs = plt.subplots(len(trained_keys), 3, figsize=(6.6, 11.0))
    for r, (name, k) in enumerate(trained_keys):
        for c, l in enumerate((0, 5, 11)):
            im = heat(axs[r, c], maps[f'trained|{name}'][l].mean(0), f'layer {l}' if r == 0 else None)
            if c == 0:
                axs[r, c].set_ylabel(name.replace(' final', '') + '\nquery position', fontsize=8)
            if r == len(trained_keys) - 1:
                axs[r, c].set_xlabel('key position', fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.35, label='attention weight, head mean (clipped at 0.5)')
    save(fig, F, 'fig_attention_trained')

    # 10. self-weight per head with the bias forced to 0, during annealing (learned logits only)
    steps = [s for s in [500, 750, 1000, 1250] if f'approach_2 step {s}' in tr and f'approach_4 step {s}' in tr]
    if steps:
        fig, axs = plt.subplots(2, len(steps), figsize=(2.3 * len(steps) + 1, 4.6))
        axs = np.atleast_2d(axs)
        for r, k in enumerate(['approach_2', 'approach_4']):
            for c, s in enumerate(steps):
                d = tr[f'{k} step {s}']
                im = axs[r, c].imshow(np.array(d['self_w_head_nobias']), cmap='Blues', vmin=0, vmax=1, aspect='auto')
                axs[r, c].set_xticks([0, 11]); axs[r, c].set_yticks([0, 11]); axs[r, c].grid(False)
                if r == 0:
                    axs[r, c].set_title(f'step {s} (b={d["diag_bias_now"]:.1f})', fontsize=8.5)
                if c == 0:
                    axs[r, c].set_ylabel(f'{k}\nlayer', fontsize=8.5)
                if r == 1:
                    axs[r, c].set_xlabel('head', fontsize=8)
        fig.colorbar(im, ax=axs, shrink=0.6, label='$A_{ii}$ with the bias removed')
        save(fig, F, 'fig_selfweight_learned')

    # numbers.tex
    def at(k, s):
        return next(r['val_loss'] for r in logs[k][0] if r['step'] == s and not r.get('final'))
    gn = lambda k, a, b: np.median([r['grad_norm'] for r in logs[k][1] if a <= r['step'] < b])
    nums.update({
        'Unigram': f'{UNIGRAM:.3f}',
        'FinalATwo': f"{fin['approach_2']:.3f}", 'FinalAFour': f"{fin['approach_4']:.3f}",
        'FinalBase': f"{fin['baseline']:.3f}", 'FinalTR': f"{fin['three_residual']:.3f}", 'FinalAThree': f"{fin['approach_3']:.3f}",
        'GapATwo': f"{fin['approach_2'] - fin['baseline']:.3f}",
        'BestAOne': f"{a1best['val_loss']:.2f}", 'BestAOneStep': str(a1best['step']),
        'AOneAtThreeK': f"{at('approach_1', 3000):.2f}", 'AOneAtThreeQK': f"{at('approach_1', 3250):.2f}",
        'AFourAtSevenFifty': f"{at('approach_4', 750):.2f}", 'AFourAtOneK': f"{at('approach_4', 1000):.2f}",
        'AFourAtOneTwoFiveK': f"{at('approach_4', 1250):.2f}",
        'ATwoAtOneK': f"{at('approach_2', 1000):.2f}", 'ATwoAtOneTwoFiveK': f"{at('approach_2', 1250):.2f}",
        'AOneAtOneK': f"{at('approach_1', 1000):.2f}", 'AOneAtOneTwoFiveK': f"{at('approach_1', 1250):.2f}",
        'TRAtOneK': f"{at('three_residual', 1000):.2f}", 'BaseAtOneK': f"{at('baseline', 1000):.2f}",
        'GnAOneEarly': f"{gn('approach_1', 1000, 2500):.2f}", 'GnAOneLate': f"{gn('approach_1', 3000, 3500):.1f}",
        'GnATwoLate': f"{gn('approach_2', 3000, 3500):.2f}", 'GnTRLate': f"{gn('three_residual', 3000, 3500):.2f}",
        'GnBaseLate': f"{gn('baseline', 3000, 3500):.2f}",
        'GnAFourMid': f"{gn('approach_4', 650, 700):.2f}", 'GnAFourEnd': f"{gn('approach_4', 1000, 1050):.1f}",
    })
    for k in NEW:
        w = {'1': 'One', '2': 'Two', '4': 'Four'}[k[-1]]
        nums[f'NoBias{w}'] = f"{curve(nobias[k][0])[1][-1]:.2f}"
        nums[f'ProbeZero{w}'] = f"{probe[f'{k}|b0']['rank1'][-1]:.3f}"
        nums[f'ProbeEight{w}'] = f"{probe[f'{k}|b8']['rank1'][-1]:.3f}"
    nums['ProbeBase'] = f"{probe['baseline|b0']['rank1'][-1]:.3f}"
    for k in ['approach_2', 'approach_4']:
        for s in [750, 1000]:
            d = tr.get(f'{k} step {s}')
            if d:
                tag = {'approach_2': 'ATwo', 'approach_4': 'AFour'}[k] + {750: 'SevenFifty', 1000: 'OneK'}[s]
                nums[f'Vb{tag}'] = f"{d['val_loss_bias']:.2f}"; nums[f'Vz{tag}'] = f"{d['val_loss_nobias']:.2f}"
                nums[f'Sz{tag}'] = f"{np.mean(d['self_w_nobias']):.2f}"
    for name, k in trained_keys:
        tag = {'baseline': 'Base', 'three_residual': 'TR', 'approach_1': 'AOne', 'approach_2': 'ATwo', 'approach_4': 'AFour'}[k]
        nums[f'Self{tag}'] = f"{np.mean(tr[name]['self_w']):.2f}"
        nums[f'Rank{tag}'] = f"{tr[name]['rank1'][-1]:.3f}"
        nums[f'SelfHeads{tag}'] = str(int((np.array(tr[name]['self_w_head']) > 0.5).sum()))
    nums['TokATwo'] = f"{np.median([r['tok_per_s'] for r in logs['approach_2'][1][10:]]) / 1e3:.0f}"
    nums['TokAFour'] = f"{np.median([r['tok_per_s'] for r in logs['approach_4'][1][10:]]) / 1e3:.0f}"
    with open(os.path.join(args.dir, 'numbers.tex'), 'w') as f:
        for k, v in nums.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(nums, open(os.path.join(D, 'summary.json'), 'w'), indent=1)
    print(json.dumps(nums, indent=1))


if __name__ == '__main__':
    main()
