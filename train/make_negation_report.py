"""Figures and LaTeX tables for the 2026-10-21 negation-prompt report (pretrained models and the negation fine-tune).

Reads <report>/data/{prompts_pretrained.json, ft_*_results.json, ft_*_log.jsonl}; writes figures/ and tables.tex.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REP = sys.argv[1] if len(sys.argv) > 1 else 'experiment_log/2026-10-21_negation_prompts'
D, F = os.path.join(REP, 'data'), os.path.join(REP, 'figures')
os.makedirs(F, exist_ok=True)

INK, INK2, GRID = '#1f1e1d', '#52514e', '#e4e2dc'
MODELS = [  # key in prompts_pretrained.json, label, color
    ('openai_gpt2_124M', 'OpenAI GPT-2 124M', '#52514e'),
    ('baseline_diag8_anneal1000', 'GPT-2 + bias (ours)', '#2a78d6'),
    ('approach_3', 'approach_3', '#eda100'),
    ('tr_diag8_anneal1000', 'three_residual', '#eb6834'),
    ('approach_2_diag8_anneal1000', 'approach_2', '#e87ba4'),
    ('approach_1_diag8_anneal1000', 'approach_1 (step 3000)', '#1baf7a'),
    ('approach_4_diag8_anneal1000', 'approach_4', '#008300'),
]
# per prompt: short label, correct answer (None = no single right answer), city named in the prompt (copy target)
PROMPTS = [('P1', 'Paris', None), ('P2', 'Paris', 'Chicago'), ('P3', 'Paris', 'Chicago'),
           ('P4', None, 'Paris'), ('P5', 'Paris', 'Berlin')]
C_OK, C_COPY = '#2a78d6', '#d0392b'

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(F, f'{name}.{ext}'), bbox_inches='tight', dpi=200)
    plt.close(fig)


def tex(s):
    return (s.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
            .replace('#', '\\#').replace('$', '\\$'))


def show(s, n=14):
    """Greedy text for a table cell: first line only, at most n characters."""
    s = s.strip(' ').split('\n')[0]
    return tex(s[:n] + ('...' if len(s) > n else ''))


pre = json.load(open(os.path.join(D, 'prompts_pretrained.json')))
ft = {n: json.load(open(os.path.join(D, f'ft_{n}_results.json'))) for n in ('approach_2', 'baseline')}
FTLAB = {'approach_2': ('approach_2', '#e87ba4'), 'baseline': ('GPT-2 + bias (ours)', '#2a78d6')}

# ---------- Fig 1: pretrained models, P(correct) and P(copied city) per prompt ----------
fig, axes = plt.subplots(1, 5, figsize=(11, 3.4), sharey=True)
ys = np.arange(len(MODELS))[::-1]
for ax, (pl, ok, cp), i in zip(axes, PROMPTS, range(5)):
    for y, (k, lab, col) in zip(ys, MODELS):
        c = pre[k][i]['candidates']
        if ok:
            ax.scatter(max(c[ok][0], 1e-5), y, s=34, color=C_OK, zorder=3, edgecolor='white', linewidth=1.2)
        if cp:
            ax.scatter(max(c[cp][0], 1e-5), y, s=34, color=C_COPY, marker='s', zorder=3, edgecolor='white',
                       linewidth=1.2)
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1.5)
    ax.set_xticks([1e-4, 1e-2, 1])
    ax.set_xticklabels(['0.0001', '0.01', '1'])
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_title(pl + ('' if not ok else f'  ({ok} correct)') if ok else pl + '  (no single answer)', fontsize=9,
                 color=INK)
    ax.set_xlabel('next-token probability')
axes[0].set_yticks(ys)
axes[0].set_yticklabels([m[1] for m in MODELS])
for t, (_, _, col) in zip(axes[0].get_yticklabels(), MODELS):
    t.set_color(INK)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker='o', ls='', color=C_OK, label='correct city (Paris)'),
                    Line2D([], [], marker='s', ls='', color=C_COPY, label='city named in the prompt')],
           loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.07))
fig.tight_layout()
save(fig, 'fig_pretrained_probs')

# ---------- Fig 2: fine-tune, held-out accuracy and distractor rate before/after ----------
FORMS = [('plain', 'plain'), ('neg', '"not in X" once'), ('rep', '"not in X" 2-6 times')]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
w = 0.2
for ax, metric, title in [(axes[0], 'acc', 'held-out facts: answer correct'),
                          (axes[1], 'distractor_rate', 'held-out facts: answers the negated city')]:
    x = np.arange(len(FORMS))
    for j, (n, ph) in enumerate([(n, ph) for n in ('baseline', 'approach_2') for ph in ('before', 'after')]):
        lab, col = FTLAB[n]
        vals = [ft[n][ph]['heldout'][f][metric] for f, _ in FORMS]
        ax.bar(x + (j - 1.5) * w, vals, w * 0.9, color=col, alpha=1.0 if ph == 'after' else 0.35,
               label=f'{lab}, {ph}')
        for xi, v in zip(x + (j - 1.5) * w, vals):
            ax.text(xi, v + 0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f[1] for f in FORMS])
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9.5, color=INK)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
save(fig, 'fig_ft_heldout')

# ---------- Fig 3: fine-tune, test prompts before/after ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
for ax, n in zip(axes, ('approach_2', 'baseline')):
    lab, col = FTLAB[n]
    for i, (pl, ok, cp) in enumerate(PROMPTS):
        for ph, off in (('before', -0.13), ('after', 0.13)):
            p = ft[n][ph]['test_prompts'][i]['p']
            a = 0.35 if ph == 'before' else 1.0
            if ok:
                ax.scatter(i + off, max(p[ok], 1e-5), color=C_OK, s=36, alpha=a, zorder=3, edgecolor='white')
            if cp:
                ax.scatter(i + off, max(p[cp], 1e-5), color=C_COPY, marker='s', s=36, alpha=a, zorder=3,
                           edgecolor='white')
        ax.text(i - 0.13, 1.6e-5, 'bef.', ha='center', fontsize=6.5, color=INK2)
        ax.text(i + 0.13, 1.6e-5, 'aft.', ha='center', fontsize=6.5, color=INK2)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 2)
    ax.set_xticks(range(5))
    ax.set_xticklabels([p[0] for p in PROMPTS])
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_title(f'{lab}: before (faded) and after fine-tuning', fontsize=9.5, color=INK)
axes[0].set_ylabel('next-token probability')
fig.legend(handles=[Line2D([], [], marker='o', ls='', color=C_OK, label='Paris (correct for P1-P3, P5)'),
                    Line2D([], [], marker='s', ls='', color=C_COPY, label='city named in the prompt')],
           loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
save(fig, 'fig_ft_prompts')

# ---------- Fig 4: fine-tune curves ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
for n in ('baseline', 'approach_2'):
    lab, col = FTLAB[n]
    L = [json.loads(l) for l in open(os.path.join(D, f'ft_{n}_log.jsonl'))]
    s = np.array([r['step'] for r in L])
    ft_l = np.array([r['loss_ft'] for r in L])
    lm = np.array([r['loss_lm'] for r in L])
    k = 20
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode='valid')
    axes[0].plot(s, ft_l, color=col, lw=0.6, alpha=0.3)
    axes[0].plot(s[k - 1:], sm(ft_l), color=col, lw=2, label=lab)
    axes[1].plot(s, lm, color=col, lw=0.6, alpha=0.3)
    axes[1].plot(s[k - 1:], sm(lm), color=col, lw=2, label=lab)
axes[0].set_yscale('log')
axes[0].set_title('answer loss on the synthetic prompts (train entities)', fontsize=9.5, color=INK)
axes[1].set_title('OpenWebText loss on the replay batches', fontsize=9.5, color=INK)
for ax in axes:
    ax.set_xlabel('fine-tuning step')
    ax.grid(color=GRID, lw=0.6)
axes[0].legend(frameon=False)
fig.tight_layout()
save(fig, 'fig_ft_curves')

# ---------- tables ----------
out = []
out.append('% generated by train/make_negation_report.py')
# pretrained greedy outputs
rows = []
for k, lab, _ in MODELS:
    cells = [show(pre[k][i]['greedy']) for i in range(5)]
    rows.append(tex(lab) + ' & ' + ' & '.join('\\texttt{' + c + '}' for c in cells) + ' \\\\')
out.append('\\newcommand{\\PretrainedGreedyRows}{' + '\n'.join(rows) + '}')
# pretrained rank of Paris / copy city
rows = []
for k, lab, _ in MODELS:
    cells = []
    for i, (pl, ok, cp) in enumerate(PROMPTS):
        c = pre[k][i]['candidates']
        okc = f'{c[ok][0]:.3f} (\\#{c[ok][1]})' if ok else '--'
        cpc = f'{c[cp][0]:.3f} (\\#{c[cp][1]})' if cp else '--'
        cells += [okc, cpc]
    rows.append(tex(lab) + ' & ' + ' & '.join(cells) + ' \\\\')
out.append('\\newcommand{\\PretrainedProbRows}{' + '\n'.join(rows) + '}')
# fine-tune greedy before/after
rows = []
for n in ('approach_2', 'baseline'):
    for ph in ('before', 'after'):
        cells = [show(t['greedy']) for t in ft[n][ph]['test_prompts']]
        rows.append(tex(FTLAB[n][0]) + f', {ph} & ' + ' & '.join('\\texttt{' + c + '}' for c in cells)
                    + ' \\\\')
    if n == 'approach_2':
        rows.append('\\midrule')
out.append('\\newcommand{\\FtGreedyRows}{' + '\n'.join(rows) + '}')
# fine-tune held-out / train-entity metrics
rows = []
for n in ('approach_2', 'baseline'):
    for ph in ('before', 'after'):
        r = ft[n][ph]
        h, t = r['heldout'], r['train_entities']
        cells = [f"{h[f]['acc']:.0%}" for f in ('plain', 'neg', 'rep')]
        cells += [f"{h[f]['distractor_rate']:.0%}" for f in ('neg', 'rep')]
        cells += [f"{t['neg']['acc']:.0%}", f"{r['owt_val_loss']:.3f}"]
        rows.append(tex(FTLAB[n][0]) + f', {ph} & ' + ' & '.join(c.replace('%', '\\%') for c in cells) + ' \\\\')
    if n == 'approach_2':
        rows.append('\\midrule')
out.append('\\newcommand{\\FtMetricRows}{' + '\n'.join(rows) + '}')
sp = ft['approach_2']['split']
out.append('\\newcommand{\\NHeldLand}{%d}\\newcommand{\\NHeldCap}{%d}\\newcommand{\\NTrainLand}{%d}'
           '\\newcommand{\\NTrainCap}{%d}' % (len(sp['landmark']['heldout']), len(sp['capital']['heldout']),
                                              len(sp['landmark']['train']), len(sp['capital']['train'])))
out.append('\\newcommand{\\HeldLandList}{' + tex(', '.join(sp['landmark']['heldout'])) + '}')
out.append('\\newcommand{\\HeldCapList}{' + tex(', '.join(sp['capital']['heldout'])) + '}')
open(os.path.join(REP, 'tables.tex'), 'w').write('\n'.join(out) + '\n')
print('done')
