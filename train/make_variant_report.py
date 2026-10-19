"""Figures, numbers.tex and summary.json for the attention-residual variants report (baseline + approach_1..5)."""
import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = ['baseline', 'approach_1', 'approach_2', 'approach_3', 'approach_4', 'approach_5']
RUN = {'baseline': 'baseline', 'approach_1': 'approach_1', 'approach_2': 'approach_2', 'approach_3': 'approach_3',
       'approach_4': 'approach_4_full', 'approach_5': 'approach_5_a0.9'}
LABEL = {'baseline': 'baseline (GPT-2)', 'approach_1': 'approach_1', 'approach_2': 'approach_2',
         'approach_3': 'approach_3', 'approach_4': 'approach_4', 'approach_5': 'approach_5 (α=0.9)'}
# reference palette slots 1-6, in fixed order
COLOR = {'baseline': '#2a78d6', 'approach_1': '#eb6834', 'approach_2': '#1baf7a', 'approach_3': '#eda100',
         'approach_4': '#e87ba4', 'approach_5': '#008300'}
COLLAPSED = {'approach_1', 'approach_2', 'approach_4'}
LS = {n: ('--' if n in COLLAPSED else '-') for n in NAMES}
MK = {'baseline': 'o', 'approach_1': 's', 'approach_2': 'D', 'approach_3': '^', 'approach_4': 'v', 'approach_5': 'P'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
TRAINED = ['baseline', 'approach_3', 'approach_5']

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 1.8,
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


def fig_final_bars(me, out):
    v20 = me['val20M']
    rows = sorted([(n, v20[n]) for n in NAMES], key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for i, (n, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=COLOR[n], height=0.6)
        ax.text(v + 0.05, y, f'{v:.3f}', va='center', fontsize=8, color=INK2)
        ax.text(-0.08, y, LABEL[n], ha='right', va='center', fontsize=8.5, color=INK, transform=ax.get_yaxis_transform())
    ax.axvline(v20['openai_gpt2_124M'], color=INK2, ls='--', lw=1)
    ax.text(v20['openai_gpt2_124M'], len(rows) - 0.35, f'OpenAI GPT-2 124M\n(pretrained) {v20["openai_gpt2_124M"]:.3f}',
            fontsize=7.5, color=INK2, ha='right', va='bottom')
    ax.axvline(v20['unigram'], color=MUTED, ls=':', lw=1.2)
    ax.text(v20['unigram'], len(rows) - 0.35, f'unigram\n{v20["unigram"]:.3f}', fontsize=7.5, color=INK2,
            ha='center', va='bottom')
    ax.set(yticks=[], xlim=(0, 10.6), xlabel='validation loss, full 20M-token split (nats/token, lower is better)')
    ax.grid(axis='y', visible=False)
    save(fig, out, 'fig_final_loss')


def fig_val(logs, me, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={'width_ratios': [1, 1.1]})
    for n in NAMES:
        val = [r for r in logs[n][0] if not r.get('final')]
        x = [r['tokens'] / 1e9 for r in val]
        y = [r['val_loss'] for r in val]
        a.plot(x, y, ls=LS[n], marker=MK[n], ms=3.5, color=COLOR[n], label=LABEL[n])
        if n in TRAINED:
            k = [i for i, r in enumerate(val) if r['step'] >= 1000]
            b.plot([x[i] for i in k], [y[i] for i in k], ls='-', marker=MK[n], ms=3.5, color=COLOR[n],
                   label=f'{LABEL[n]}: {y[-1]:.3f} at 2.49B')
    for ax, ref in ((a, True), (b, True)):
        ax.axhline(me['val4M']['openai_gpt2_124M'], color=INK2, ls='--', lw=1)
    a.axhline(me['val4M']['unigram'], color=MUTED, ls=':', lw=1.2)
    a.text(2.6, me['val4M']['unigram'] + 0.12, 'unigram (no context)', ha='right', fontsize=7.5, color=INK2)
    a.text(2.6, me['val4M']['openai_gpt2_124M'] - 0.12, 'OpenAI GPT-2 124M (pretrained)', ha='right', va='top',
           fontsize=7.5, color=INK2)
    a.set(xlabel='training tokens (billions)', ylabel='validation loss (4.2M-token subset)', ylim=(2.8, 8.4),
          xlim=(-0.05, 2.65), title='All variants')
    a.legend(loc='center right', bbox_to_anchor=(1.0, 0.42), ncol=1)
    b.text(0.55, me['val4M']['openai_gpt2_124M'] - 0.008, 'OpenAI GPT-2 124M', fontsize=7.5, color=INK2, va='top')
    b.set(xlabel='training tokens (billions)', xlim=(0.45, 2.6), ylim=(3.15, 4.05),
          title='Variants that train, from 0.5B tokens (zoom)')
    b.legend(loc='upper right')
    fig.tight_layout()
    save(fig, out, 'fig_val_loss')


def fig_gap(logs, out):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    base = {r['step']: r['val_loss'] for r in logs['baseline'][0] if not r.get('final')}
    ax.axhspan(-0.01, 0.01, color=GRID, alpha=0.8, lw=0)
    ax.text(1.3, -0.02, 'shaded: assumed seed-to-seed spread (±0.01)', fontsize=7.5, color=INK2, ha='center', va='top')
    ax.axhline(0, color=COLOR['baseline'], lw=1.2)
    ax.text(0.02, 0.006, 'baseline', color=INK2, fontsize=7.5, transform=ax.get_yaxis_transform())
    for n in ['approach_3', 'approach_5']:
        val = [r for r in logs[n][0] if not r.get('final') and r['step'] > 0]
        x = [r['tokens'] / 1e9 for r in val]
        y = [r['val_loss'] - base[r['step']] for r in val]
        ax.plot(x, y, marker=MK[n], ms=4, color=COLOR[n], label=LABEL[n])
        ax.annotate(f'{y[-1]:+.3f}', (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points', fontsize=8,
                    color=INK2, va='center')
    ax.set(xlabel='training tokens (billions)', ylabel='val loss − baseline val loss (nats)', xlim=(0, 2.75), ylim=(-0.04, 0.3),
           title='Gap to the baseline at matched steps (positive = worse)')
    ax.legend(loc='upper right')
    save(fig, out, 'fig_gap_to_baseline')


def fig_train(logs, uni, out):
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    for n in NAMES:
        tr = logs[n][1]
        s = [r['step'] for r in tr]
        a.plot(s, ema([r['train_loss'] for r in tr]), ls=LS[n], color=COLOR[n], label=LABEL[n])
        b.plot(s, ema([r['grad_norm'] for r in tr]), ls=LS[n], color=COLOR[n], lw=1.2)
    a.axhline(uni, color=MUTED, ls=':', lw=1.2)
    a.set(ylabel='train loss (EMA)', ylim=(2.9, 11.2), title='Training loss')
    a.legend(loc='upper right', ncol=2)
    b.set(ylabel='grad norm (EMA, pre-clip)', yscale='log', xlabel='step (524,288 tokens / step)',
          title='Global gradient norm')
    fig.tight_layout()
    save(fig, out, 'fig_train_dynamics')


def fig_alpha(sweep, probes, out, T=1024, D=64, L=12):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    # (a) real models at init: rank distance through depth for several alpha (sequential single hue)
    shades = {0.0: '#c6dbef', 0.5: '#9ecae1', 0.75: '#6baed6', 0.9: '#3182bd', 0.95: '#1c5fa8', 1.0: '#08306b'}
    by_alpha = {s['alpha']: s['rank1_by_layer'] for s in sweep}
    for al, col in shades.items():
        c = by_alpha[al]
        a.plot(range(-1, L), c, '-o', ms=3, color=col)
        a.annotate(f'α={al:g}', (L - 1, c[-1]), xytext=(5, 0), textcoords='offset points', fontsize=7.5,
                   color=INK2, va='center')
    a.set(xlabel='layer', ylabel='distance from rank 1', ylim=(0, 1.05), xlim=(-1.3, 12.6),
          xticks=[-1, 0, 3, 6, 9, 11], xticklabels=['emb', 0, 3, 6, 9, 11],
          title='approach_5 at initialization: collapse through depth')
    # (b) real models at init: layer-11 rank distance vs alpha, plus each variant's init value
    al = [s['alpha'] for s in sweep]
    r = [s['rank1_L11'] for s in sweep]
    b.plot(al, r, '-', color=MUTED, lw=1.4, zorder=1)
    b.scatter(al, r, s=14, color=MUTED, zorder=2, label='approach_5 at init, α sweep')
    pts = {'approach_1': 0.0, 'approach_2': 0.0, 'approach_4': 0.0, 'approach_5': 0.9, 'approach_3': 1.0,
           'baseline': 1.0}
    offs = {'approach_1': (14, 30), 'approach_2': (14, 18), 'approach_4': (14, 6), 'approach_5': (-10, 12),
            'approach_3': (-12, -14), 'baseline': (-12, 8)}
    for n, x in pts.items():
        y = probes[f'{n}@init']['layers'][-1]['rank1_res']
        b.scatter([x], [y], s=46, marker=MK[n], color=COLOR[n], edgecolor='white', linewidth=0.8, zorder=3)
        fate = 'collapsed' if n in COLLAPSED else 'trains'
        b.annotate(f'{LABEL[n]} → {fate}', (x, y), xytext=offs[n], textcoords='offset points', fontsize=7.5,
                   color=INK2, ha='left' if x < 0.5 else 'right', va='center')
    b.axvspan(-0.02, 0.8, color='#e34948', alpha=0.06, lw=0)
    b.text(0.4, 0.62, 'near-collapsed at init\n(distance ≤ 0.2)', ha='center', fontsize=7.5, color=INK2)
    b.set(xlabel='α (weight of the unmixed skip X)', ylabel='distance from rank 1 at layer 11', ylim=(0, 0.8),
          xlim=(-0.05, 1.05), title='Real 12-layer models at initialization')
    b.legend(loc='upper left', bbox_to_anchor=(0.0, 0.98))
    fig.tight_layout()
    save(fig, out, 'fig_alpha_and_collapse')


def fig_layers(probes, out):
    keys = [('cos', 'cosine between positions', (0, 1.05)), ('rank1_res', 'distance from rank 1', (0, 1.05)),
            ('ent_ratio', 'attention entropy / uniform', (0, 1.05)), ('self_w', 'attention on own position', None)]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6), sharex=True)
    for row, tag in enumerate(['init', 'trained']):
        for ax, (key, title, ylim) in zip(axes[row], keys):
            for n in NAMES:
                k = f'{n}@{tag}'
                if k not in probes:
                    continue
                rows = [r for r in probes[k]['layers'] if key in r]
                xs = [-1 if r['layer'] == 'emb' else r['layer'] for r in rows]
                ax.plot(xs, [r[key] for r in rows], ls=LS[n], marker=MK[n], ms=3, color=COLOR[n], lw=1.4)
            ax.set(title=f'{title} — {"initialization" if tag == "init" else "trained"}')
            if ylim:
                ax.set_ylim(*ylim)
            if row == 1:
                ax.set(xlabel='layer', xticks=[-1, 0, 3, 6, 9, 11], xticklabels=['emb', 0, 3, 6, 9, 11])
    handles = [plt.Line2D([], [], color=COLOR[n], ls=LS[n], marker=MK[n], ms=4, label=LABEL[n]) for n in NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.075, 'Trained approach_1/2 are their step-3000 checkpoints; approach_4 and the others are '
             'their final (step-4768) checkpoints.', ha='center', fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, out, 'fig_layer_diagnostics')


def fig_heat(heat_path, out):
    h = np.load(heat_path)
    names = ['baseline', 'approach_1', 'approach_3', 'approach_4', 'approach_5']
    fig, axes = plt.subplots(len(names), 3, figsize=(7.0, 11.4))
    for i, n in enumerate(names):
        for j, l in enumerate([0, 5, 11]):
            ax = axes[i, j]
            im = ax.imshow(h[f'{n}|L{l}'], cmap='Blues', vmin=0, vmax=0.5, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(f'layer {l}')
            if j == 0:
                ax.set_ylabel(LABEL[n], color=INK2)
    fig.suptitle('Trained attention, mean over 12 heads, first 48 tokens of a val sequence\n'
                 '(approach_2 is identical to approach_1)', fontsize=9)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.4, pad=0.02)
    cb.set_label('attention weight (clipped at 0.5)', color=INK2)
    save(fig, out, 'fig_attention_heatmaps')


def fig_mechanism(probes, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    L = list(range(12))
    lay = lambda k: [r for r in probes[k]['layers'] if r['layer'] != 'emb']
    # (a) approach_3: size of the new term vs the standard attention term
    a3 = lay('approach_3@trained')
    axes[0].plot(L, [r['n_attnV'] for r in a3], '-o', ms=3, color=COLOR['approach_3'], label='|Concat[A V] W_O|')
    axes[0].plot(L, [r['n_mixX_Wo'] for r in a3], '--^', ms=3, color=INK2, label='|Concat[A X_h] W_O|  (new term)')
    axes[0].set(xlabel='layer', ylabel='mean L2 norm per token', title='approach_3: the new term is used')
    axes[0].legend(loc='upper left')
    # (b) self-attention weight: approach_5 vs baseline vs approach_3
    for n in ['baseline', 'approach_3', 'approach_5']:
        axes[1].plot(L, [r['self_w'] for r in lay(f'{n}@trained')], '-', marker=MK[n], ms=3.5, color=COLOR[n],
                     label=LABEL[n])
    axes[1].set(xlabel='layer', ylabel='mean attention on own position',
                title='Self-attention moves to layers 8-10 in approach_5')
    axes[1].legend(loc='upper left')
    # (c) residual norm growth
    for n in ['baseline', 'approach_3', 'approach_5']:
        rows = probes[f'{n}@trained']['layers']
        axes[2].plot([-1] + L, [r['norm'] for r in rows], '-', marker=MK[n], ms=3.5, color=COLOR[n], label=LABEL[n])
    axes[2].set(xlabel='layer', ylabel='residual-stream norm', xticks=[-1, 0, 3, 6, 9, 11],
                xticklabels=['emb', 0, 3, 6, 9, 11], title='α=0.9 flattens residual-norm growth')
    axes[2].legend(loc='upper left')
    fig.tight_layout()
    save(fig, out, 'fig_mechanism_trained')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data, figs = os.path.join(args.out, 'data'), os.path.join(args.out, 'figures')
    os.makedirs(figs, exist_ok=True)
    logs = {n: load_log(os.path.join(args.runs, RUN[n], 'log.jsonl')) for n in NAMES}
    probes = json.load(open(os.path.join(data, 'variants_probe.json')))
    sweep = json.load(open(os.path.join(data, 'alpha_sweep.json')))
    me = json.load(open(os.path.join(data, 'matched_evals.json')))

    fig_final_bars(me, figs)
    fig_val(logs, me, figs)
    fig_gap(logs, figs)
    fig_train(logs, me['val4M']['unigram'], figs)
    fig_alpha(sweep, probes, figs)
    fig_layers(probes, figs)
    fig_heat(os.path.join(data, 'heatmaps.npz'), figs)
    fig_mechanism(probes, figs)

    # copy run logs and configs into the report folder
    for n in NAMES:
        d = os.path.join(data, n)
        os.makedirs(d, exist_ok=True)
        for f in ('log.jsonl', 'config.json'):
            src = os.path.join(args.runs, RUN[n], f)
            if os.path.exists(src):
                shutil.copy(src, d)

    # numbers.tex
    M = {}
    pre = {'baseline': 'Base', 'approach_1': 'One', 'approach_2': 'Two', 'approach_3': 'Three',
           'approach_4': 'Four', 'approach_5': 'Five'}
    for n in NAMES:
        p = pre[n]
        val, train = logs[n]
        M[f'{p}Steps'] = str(train[-1]['step'] + 1)
        M[f'{p}Tok'] = f'{(train[-1]["step"] + 1) * 524288 / 1e9:.2f}B'
        for s in (250, 500, 1000, 2000, 3000):
            r = next((r for r in val if r['step'] == s and not r.get('final')), None)
            M[f'{p}At{ {250: "A", 500: "B", 1000: "C", 2000: "D", 3000: "E"}[s] }'] = f'{r["val_loss"]:.3f}' if r else '--'
        M[f'{p}Twenty'] = f'{me["val20M"][n]:.3f}' if n in me['val20M'] else '--'
        M[f'{p}Four'] = f'{me["val4M"][n]:.3f}'
        if n in me['val20M']:
            M[f'{p}Delta'] = f'{me["val20M"][n] - me["val20M"]["baseline"]:+.3f}'
        init = probes[f'{n}@init']['layers'][-1]
        M[f'{p}InitRank'] = f'{init["rank1_res"]:.3f}'
        M[f'{p}InitCos'] = f'{init["cos"]:.3f}'
        if f'{n}@trained' in probes:
            t = probes[f'{n}@trained']
            M[f'{p}TrRank'] = f'{t["layers"][-1]["rank1_res"]:.3f}'
            M[f'{p}KL'] = f'{t["kl_to_unigram"]:.3f}'
    M['RefTwenty'] = f'{me["val20M"]["openai_gpt2_124M"]:.3f}'
    M['RefFour'] = f'{me["val4M"]["openai_gpt2_124M"]:.3f}'
    M['UniTwenty'] = f'{me["val20M"]["unigram"]:.3f}'
    M['UniFour'] = f'{me["val4M"]["unigram"]:.3f}'
    M['ThreeDelta'] = f'{me["val20M"]["approach_3"] - me["val20M"]["baseline"]:+.3f}'
    M['FiveDelta'] = f'{me["val20M"]["approach_5"] - me["val20M"]["baseline"]:+.3f}'
    with open(os.path.join(args.out, 'numbers.tex'), 'w') as f:
        f.write('% auto-generated by train/make_variant_report.py\n')
        for k, v in M.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump({'matched_evals': me, 'steps_trained': {n: int(M[f"{pre[n]}Steps"]) for n in NAMES}},
              open(os.path.join(data, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
