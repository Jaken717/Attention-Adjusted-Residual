"""Measurements for the 2026-10-20 report: approach_1, approach_2 and approach_4 trained with the annealed diagonal bias.

Writes to --out:
  probe_init.json   rank-1 distance and self-weight per layer at init, with and without the bias, per variant
  trained.json      the same for trained checkpoints, plus (for the annealing checkpoints) val loss with the
                    bias as stored and with the bias forced to 0
  attention.npz     attention maps (first N tokens of val sequence 0), per model, layer and head
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from three_residual_analysis import build, measure, T

N_EVAL = 2 * 1024 * 1024  # val tokens for the bias-on / bias-off loss comparison


@torch.no_grad()
def val_loss(m, val, n_tokens=N_EVAL, B=16):
    tot, k = 0.0, 0
    for i in range(0, n_tokens, B * T):
        buf = torch.from_numpy(val[i:i + B * T + 1].astype(np.int64)).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = m(buf[:-1].view(B, T))
        tot += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), buf[1:].reshape(-1)).item()
        k += 1
    return tot / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/local/scratch/a/liu3170/data/openwebtext')
    ap.add_argument('--runs', default='/local/scratch/a/liu3170/runs/openwebtext')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seq', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    val = np.memmap(os.path.join(args.data, 'val.bin'), dtype=np.uint16, mode='r')
    idx = torch.from_numpy(val[:args.n_seq * T].astype(np.int64)).view(args.n_seq, T).cuda()
    R = args.runs

    # 1. init probe
    probe, maps = {}, {}
    for v in ['baseline', 'approach_1', 'approach_2', 'approach_4', 'three_residual']:
        for b in [0.0, 8.0]:
            m, _ = build(v, diag_bias=b)
            r = measure(m, idx, keep_maps=True)
            maps[f'init|{v}|b{b:g}'] = np.stack(r.pop('maps'))
            probe[f'{v}|b{b:g}'] = r
            print(f'init {v:14s} b={b:g} rank1 L11={r["rank1"][-1]:.3f} self_w L0={r["self_w"][0]:.2f}', flush=True)
    json.dump(probe, open(os.path.join(args.out, 'probe_init.json'), 'w'), indent=1)

    # 2. trained checkpoints. The bias is a persistent buffer, so each checkpoint carries its value at save time.
    models = [
        ('approach_1 step 3000', 'approach_1', f'{R}/approach_1_diag8_anneal1000/ckpt_step3000.pt'),
        ('approach_2 final', 'approach_2', f'{R}/approach_2_diag8_anneal1000/ckpt.pt'),
        ('approach_4 final', 'approach_4', f'{R}/approach_4_diag8_anneal1000/ckpt.pt'),
        ('three_residual final', 'three_residual', f'{R}/tr_diag8_anneal1000/ckpt.pt'),
        ('baseline final', 'baseline', f'{R}/baseline_diag8_anneal1000/ckpt.pt'),
    ]
    for v in ['approach_2', 'approach_4']:
        for s in [500, 750, 1000, 1250]:
            models.append((f'{v} step {s}', v, f'{R}/{v}_diag8_repro/ckpt_step{s}.pt'))
    trained = {}
    for name, v, ck in models:
        if not os.path.exists(ck):
            print('missing', ck, flush=True)
            continue
        m, step = build(v, ck, diag_bias=8.0)
        b_now = float(m.h[0].attn.diag[0])
        r = measure(m, idx, keep_maps=True)
        maps[f'trained|{name}'] = np.stack(r.pop('maps'))
        r.update(step=step, diag_bias_now=b_now, val_loss_bias=val_loss(m, val))
        m.set_diag(0.0)
        r0 = measure(m, idx)
        r0.pop('maps')
        r.update(val_loss_nobias=val_loss(m, val), self_w_nobias=r0['self_w'], rank1_nobias=r0['rank1'],
                 self_w_head_nobias=r0['self_w_head'])
        trained[name] = r
        print(f'{name:22s} step {step} b={b_now:.3f} val(b)={r["val_loss_bias"]:.3f} val(b=0)={r["val_loss_nobias"]:.3f} '
              f'rank1 L11={r["rank1"][-1]:.3f} self_w mean={np.mean(r["self_w"]):.2f} (b=0: {np.mean(r0["self_w"]):.2f})',
              flush=True)
    json.dump(trained, open(os.path.join(args.out, 'trained.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(args.out, 'attention.npz'), **maps)


if __name__ == '__main__':
    main()
