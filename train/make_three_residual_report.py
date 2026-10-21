"""Figures and numbers.tex for the three_residual (annealed diagonal bias) report on OpenWebText.

Reads the run logs and the outputs of three_residual_analysis.py; writes figures/*.pdf (+ .png) and numbers.tex.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
# same entity colors as the 2026-10-16 report where the entity is shared
C = {'baseline annealed': '#2a78d6', 'three_residual annealed': '#eb6834', 'approach_3': '#eda100',
     'three_residual collapsed': '#e87ba4'}
LS = {'baseline annealed': '-', 'three_residual annealed': '-', 'approach_3': '-', 'three_residual collapsed': '--'}
MK = {'baseline annealed': 'o', 'three_residual annealed': 's', 'approach_3': '^', 'three_residual collapsed': 'v'}
LABEL = {'baseline annealed': 'baseline + annealed bias', 'three_residual annealed': 'three_residual + annealed bias',
         'approach_3': 'approach_3', 'three_residual collapsed': 'three_residual, no bias'}
RUN = {'baseline annealed': 'baseline_diag8_anneal1000', 'three_residual annealed': 'tr_diag8_anneal1000',
       'approach_3': 'approach_3', 'three_residual collapsed': 'three_residual'}
RAMP = ['#b9d3f3', '#7aaee8', '#2a78d6', '#123f78']  # one hue, light -> dark (magnitude of the init setting)

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
    """Nudge label positions apart so neighbouring end-of-line labels do not overlap."""
    order = np.argsort(ys); out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        out[b] = max(out[b], out[a] + gap)
    return out


def heat(ax, a, title=None):
    im = ax.imshow(a, cmap='Blues', vmin=0, vmax=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(MUTED)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='/local/scratch/a/liu3170/runs/openwebtext')
    ap.add_argument('--dir', required=True, help='report directory (reads data/, writes figures/ and numbers.tex)')
    args = ap.parse_args()
    D = os.path.join(args.dir, 'data'); F = os.path.join(args.dir, 'figures')
    os.makedirs(F, exist_ok=True)
    probe = json.load(open(os.path.join(D, 'probe_init.json')))
    tr = json.load(open(os.path.join(D, 'trained.json')))
    maps = np.load(os.path.join(D, 'attention.npz'))
    uni = tr['unigram_val_ce']
    logs = {k: load_log(os.path.join(args.runs, v, 'log.jsonl')) for k, v in RUN.items()}
    short = {k: load_log(os.path.join(args.runs, v, 'log.jsonl'))[0] for k, v in
             [('learn', 'tr_diag8_learn_s250'), ('mimetic', 'tr_mimetic1.5_s250')]}
    layers = np.arange(-1, 12)

    # 1. init probe: rank-1 distance by layer
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1), sharey=True)
    for ax, title, names in [
            (axs[0], '(a) diagonal bias $b$ on the self logit',
             [('three_residual', '$b=0$'), ('diag_bias 3', '$b=3$'), ('diag_bias 5', '$b=5$'), ('diag_bias 8', '$b=8$')]),
            (axs[1], '(b) mimetic init, $W_Q=W_K=s\\cdot$orthogonal',
             [('three_residual', 'default'), ('mimetic 0.8', '$s=0.8$'), ('mimetic 1.0', '$s=1.0$'), ('mimetic 1.5', '$s=1.5$')])]:
        ax.plot(layers, probe['baseline']['rank1'], ls=':', color=MUTED, lw=1.4)
        ends = [probe[n]['rank1'][-1] for n, _ in names] + [probe['baseline']['rank1'][-1]]
        pos = spread(ends, 0.055)
        for (n, lab), col, yl in zip(names, RAMP, pos):
            ax.plot(layers, probe[n]['rank1'], '-o', ms=3.2, color=col)
            ax.text(11.3, yl, lab, color=INK2, fontsize=7.5, va='center')
        ax.text(11.3, pos[-1], 'baseline', color=INK2, fontsize=7.5, va='center')
        ax.axhline(0.15, color=INK2, lw=0.9, ls='--')
        ax.text(-1, 0.165, 'collapse line (~0.15)', fontsize=7.5, color=INK2)
        ax.set_title(title, fontsize=9.5, loc='left'); ax.set_xlabel('layer (−1 = embeddings)')
        ax.set_xlim(-1.3, 13.2); ax.set_ylim(0, 1.02); ax.set_xticks(range(-1, 12, 2))
    axs[0].set_ylabel('distance from rank 1 at init')
    save(fig, F, 'fig_init_probe')

    # 2. attention at init: no bias vs bias 8
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for r, (key, lab) in enumerate([('init|three_residual', 'no bias'), ('init|diag_bias 8', 'diagonal bias $b=8$')]):
        for c, l in enumerate((0, 5, 11)):
            im = heat(axs[r, c], maps[key][l].mean(0), f'layer {l}' if r == 0 else None)
            if c == 0:
                axs[r, c].set_ylabel(f'{lab}\nquery position', fontsize=8.5)
    fig.colorbar(im, ax=axs, shrink=0.55, label='attention weight (clipped at 0.5)')
    save(fig, F, 'fig_init_attention')

    # 3. pretraining: bias schedule, val loss, and a zoom
    fig = plt.figure(figsize=(8.4, 7.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2.3, 2.0], hspace=0.38)
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    trn = logs['three_residual annealed'][1]
    a0.plot([r['step'] for r in trn], [r['diag_bias'] for r in trn], color=INK2)
    a0.axvspan(0, 1000, color=GRID, alpha=0.6, lw=0)
    a0.text(560, 6.2, 'annealing: $b = 8\\,(1 - t/1000)$', ha='left', fontsize=8, color=INK2)
    a0.text(2900, 2.0, '$b = 0$: plain three_residual / plain baseline', ha='center', fontsize=8, color=INK2)
    a0.set_ylabel('bias $b$'); a0.set_ylim(-0.4, 8.6); a0.set_xlim(0, 4800); a0.set_yticks([0, 4, 8])
    a0.set_title('(a) diagonal-bias schedule (both annealed runs)', fontsize=9.5, loc='left')
    for ax, title in [(a1, '(b) validation loss (4M-token eval every 250 steps)'),
                      (a2, '(c) zoom: steps 1000–4750, after the bias reached 0')]:
        for k in ['three_residual collapsed', 'three_residual annealed', 'baseline annealed', 'approach_3']:
            v = [r for r in logs[k][0] if not r.get('final')]
            ax.plot([r['step'] for r in v], [r['val_loss'] for r in v], ls=LS[k], marker=MK[k], ms=3.2,
                    color=C[k], label=LABEL[k])
        ax.axvspan(0, 1000, color=GRID, alpha=0.6, lw=0)
        ax.set_title(title, fontsize=9.5, loc='left'); ax.set_ylabel('val loss')
    a1.axhline(uni, color=MUTED, ls=':', lw=1.2)
    a1.text(4750, uni + 0.12, f'unigram (no context) {uni:.2f}', ha='right', fontsize=7.5, color=INK2)
    v = logs['three_residual collapsed'][0][-1]
    a1.annotate('stopped at step 1080', (v['step'], v['val_loss']), xytext=(30, 18), textcoords='offset points',
                fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    a1.set_xlim(0, 4800); a1.set_ylim(3, 11.2); a1.legend(loc='upper right', ncol=2)
    a2.set_xlim(950, 4800); a2.set_ylim(3.25, 4.8); a2.set_xlabel('optimizer step (524,288 tokens each)')
    ks = ['three_residual annealed', 'baseline annealed', 'approach_3']
    lasts = [[r for r in logs[k][0] if not r.get('final')][-1]['val_loss'] for k in ks]
    for v, yl in zip(lasts, spread(lasts, 0.075)):
        a2.text(4790, yl, f'{v:.3f}', fontsize=7.5, color=INK2, va='center')
    save(fig, F, 'fig_training')

    # 4. short runs: val loss at step 250
    def at250(val):
        return next(r['val_loss'] for r in val if r['step'] == 250)
    rows = [('three_residual, no bias', at250(logs['three_residual collapsed'][0]), MUTED),
            ('mimetic init $s=1.5$', at250(short['mimetic']), C['three_residual annealed']),
            ('diagonal bias 8, learnable', at250(short['learn']), C['three_residual annealed']),
            ('diagonal bias 8, annealed', at250(logs['three_residual annealed'][0]), C['three_residual annealed']),
            ('approach_3 (reference)', at250(logs['approach_3'][0]), C['approach_3']),
            ('baseline + annealed bias (reference)', at250(logs['baseline annealed'][0]), C['baseline annealed'])]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for y, (lab, val, col) in enumerate(rows):
        ax.barh(y, val, color=col, height=0.6)
        ax.text(val + 0.06, y, f'{val:.2f}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.axvline(uni, color=MUTED, ls=':', lw=1.2)
    ax.text(uni + 0.08, -0.55, 'unigram', fontsize=7.5, color=INK2, ha='left')
    ax.set_xlim(0, 8.6); ax.set_xlabel('val loss at step 250 (131M tokens)'); ax.grid(axis='y', visible=False)
    save(fig, F, 'fig_short_runs')

    # 5. final loss: dot plot (differences are small, so no zero baseline)
    fin = {k: next(r['val_loss'] for r in logs[k][0] if r.get('final')) for k in
           ['three_residual annealed', 'baseline annealed', 'approach_3']}
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    order = ['approach_3', 'baseline annealed', 'three_residual annealed']
    for y, k in enumerate(order):
        ax.plot([3.3, fin[k]], [y, y], color=GRID, lw=1.2, zorder=1)
        ax.plot(fin[k], y, MK[k], ms=8, color=C[k], zorder=2)
        ax.text(fin[k] + 0.008, y + 0.22, f'{fin[k]:.3f}', fontsize=8, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels([LABEL[k] for k in order])
    ax.set_xlim(3.3, 3.68); ax.set_ylim(-0.6, 2.7); ax.set_xlabel('final val loss, full 20M-token split (lower is better)')
    ax.grid(axis='y', visible=False)
    save(fig, F, 'fig_final_loss')

    # 6. attention heatmaps: collapsed vs trained
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.4))
    for r, k in enumerate(['three_residual collapsed', 'three_residual annealed', 'baseline annealed']):
        m = tr['models'][k]
        for c, l in enumerate((0, 5, 11)):
            im = heat(axs[r, c], maps[f'trained|{k}'][l].mean(0), f'layer {l}' if r == 0 else None)
            if c == 0:
                axs[r, c].set_ylabel(f"{LABEL[k]}\nstep {m['step']}\nquery position", fontsize=8)
            if r == 2:
                axs[r, c].set_xlabel('key position', fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.45, label='attention weight (clipped at 0.5)')
    save(fig, F, 'fig_attention_trained')

    # 7. every head of the trained three_residual, layers 5 and 11
    fig, axs = plt.subplots(4, 6, figsize=(10.5, 7.4))
    sw = np.array(tr['models']['three_residual annealed']['self_w_head'])
    for blk, l in enumerate((5, 11)):
        for h in range(12):
            ax = axs[2 * blk + h // 6, h % 6]
            im = heat(ax, maps['trained|three_residual annealed'][l][h])
            ax.set_title(f'L{l} h{h}: $A_{{ii}}$={sw[l, h]:.2f}', fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.5, label='attention weight (clipped at 0.5)')
    save(fig, F, 'fig_attention_heads')

    # 8. self-weight: by layer, and by layer x head
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axs[0]
    ax.plot(range(12), probe['diag_bias 8']['self_w'], ls=':', color=MUTED, lw=1.4)
    ax.text(0, probe['diag_bias 8']['self_w'][0] + 0.03, 'all models at init with $b=8$', fontsize=7.5, color=INK2)
    for k in ['three_residual annealed', 'baseline annealed', 'approach_3', 'three_residual collapsed']:
        y = tr['models'][k]['self_w']
        ax.plot(range(12), y, ls=LS[k], marker=MK[k], ms=3.5, color=C[k], label=LABEL[k])
    ax.set_xlabel('layer'); ax.set_ylabel('mean attention on itself, $A_{ii}$'); ax.set_ylim(0, 1)
    ax.set_xticks(range(12)); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)
    ax.set_title('(a) self-weight by layer (8 val sequences, positions ≥ 64)', fontsize=9.5, loc='left')
    ax = axs[1]
    im = ax.imshow(sw, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('head'); ax.set_ylabel('layer'); ax.set_xticks(range(12)); ax.set_yticks(range(12)); ax.grid(False)
    ax.set_title('(b) three_residual + annealed bias: per head', fontsize=9.5, loc='left')
    fig.colorbar(im, ax=ax, label='$A_{ii}$')
    save(fig, F, 'fig_selfweight')

    # 9. rank-1 distance of the trained models
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for k in ['three_residual annealed', 'baseline annealed', 'approach_3', 'three_residual collapsed']:
        ax.plot(layers, tr['models'][k]['rank1'], ls=LS[k], marker=MK[k], ms=3.5, color=C[k], label=LABEL[k])
    ax.axhline(0.15, color=INK2, lw=0.9, ls='--')
    ax.set_xlabel('layer (−1 = embeddings)'); ax.set_ylabel('distance from rank 1'); ax.set_ylim(0, 1.02)
    ax.set_xticks(range(-1, 12)); ax.legend(loc='lower left', fontsize=7.5)
    save(fig, F, 'fig_rank_trained')

    # numbers.tex
    m = tr['models']
    nums = {
        'Unigram': f'{uni:.3f}',
        'FinalThree': f"{fin['three_residual annealed']:.3f}", 'FinalBaseB': f"{fin['baseline annealed']:.3f}",
        'FinalAppThree': f"{fin['approach_3']:.3f}",
        'GapBase': f"{fin['three_residual annealed'] - fin['baseline annealed']:.3f}",
        'GapAppThree': f"{fin['three_residual annealed'] - fin['approach_3']:.3f}",
        'CollapsedVal': f"{logs['three_residual collapsed'][0][-1]['val_loss']:.2f}",
        'ParamsThree': f"{m['three_residual annealed']['params']:,}".replace(',', '{,}'),
        'ParamsBase': f"{m['baseline annealed']['params']:,}".replace(',', '{,}'),
        'ProbePlain': f"{probe['three_residual']['rank1'][-1]:.3f}", 'ProbeBase': f"{probe['baseline']['rank1'][-1]:.3f}",
        'ProbeDiagEight': f"{probe['diag_bias 8']['rank1'][-1]:.3f}", 'ProbeMimetic': f"{probe['mimetic 1.5']['rank1'][-1]:.3f}",
        'SelfInitEight': f"{probe['diag_bias 8']['self_w'][0]:.2f}",
        'SelfHeadsThree': str(int((np.array(m['three_residual annealed']['self_w_head']) > 0.5).sum())),
        'SelfHeadsBase': str(int((np.array(m['baseline annealed']['self_w_head']) > 0.5).sum())),
        'SelfHeadsAppThree': str(int((np.array(m['approach_3']['self_w_head']) > 0.5).sum())),
        'SelfMeanThree': f"{np.mean(m['three_residual annealed']['self_w']):.2f}",
        'SelfMeanBase': f"{np.mean(m['baseline annealed']['self_w']):.2f}",
        'SelfMeanAppThree': f"{np.mean(m['approach_3']['self_w']):.2f}",
        'RankThree': f"{m['three_residual annealed']['rank1'][-1]:.3f}", 'RankCollapsed': f"{m['three_residual collapsed']['rank1'][-1]:.3f}",
        'ShortPlain': f'{rows[0][1]:.2f}', 'ShortMimetic': f'{rows[1][1]:.2f}', 'ShortLearn': f'{rows[2][1]:.2f}',
        'ShortAnneal': f'{rows[3][1]:.2f}', 'ShortAppThree': f'{rows[4][1]:.2f}', 'ShortBaseB': f'{rows[5][1]:.2f}',
        'ValThreeAtOneK': f"{next(r['val_loss'] for r in logs['three_residual annealed'][0] if r['step'] == 1000):.2f}",
        'ValThreeAtOneTwoFiveK': f"{next(r['val_loss'] for r in logs['three_residual annealed'][0] if r['step'] == 1250):.2f}",
        'TokThree': f"{np.median([r['tok_per_s'] for r in logs['three_residual annealed'][1][10:]]) / 1e3:.0f}",
        'TokAppThree': f"{np.median([r['tok_per_s'] for r in logs['approach_3'][1][10:]]) / 1e3:.0f}",
    }
    with open(os.path.join(args.dir, 'numbers.tex'), 'w') as f:
        for k, v in nums.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(nums, open(os.path.join(D, 'summary.json'), 'w'), indent=1)
    print(json.dumps(nums, indent=1))


if __name__ == '__main__':
    main()
