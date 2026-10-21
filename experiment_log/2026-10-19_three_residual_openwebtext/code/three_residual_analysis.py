"""Measurements for the three_residual (annealed diagonal bias) report on OpenWebText.

Writes to --out:
  probe_init.json      rank-1 distance and self-weight per layer at init, for the init settings tried
  trained.json         the same for the collapsed, annealed-trained and baseline-control checkpoints,
                       plus per-head self-weight and the unigram cross-entropy of the val split
  attention.npz        attention maps (first N tokens of val sequence 0) per model, layer and head
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from gpt2_torch import GPT, GPTConfig

SKIP = 64  # ignore the first positions, where the causal prefix is short (same as analyze_variants.py)
T, D, N = 1024, 768, 48


def rank1_distance(x):
    """||X - mean over positions|| / ||X||, averaged over sequences: 0 means every position holds the same vector."""
    xs = x[:, SKIP:].float()
    return ((xs - xs.mean(1, keepdim=True)).norm(dim=(1, 2)) / xs.norm(dim=(1, 2))).mean().item()


@torch.no_grad()
def measure(m, idx, keep_maps=False):
    mask = torch.ones(T, T, device=idx.device).tril().bool()
    eye = torch.eye(T, device=idx.device)
    x = m.wte(idx) + m.wpe(torch.arange(T, device=idx.device))
    out = {'rank1': [rank1_distance(x)], 'self_w': [], 'self_w_head': [], 'local_w': [], 'maps': []}
    for b in m.h:
        at = b.attn
        q, k = [at.split_heads(t) for t in at.c_attn(b.ln_1(x)).split(D, 2)[:2]]
        lg = q @ k.transpose(-1, -2) / (D // m.cfg.n_head) ** 0.5
        if hasattr(at, 'diag'):
            lg = lg + at.diag.view(-1, 1, 1) * eye
        A = lg.masked_fill(~mask, float('-inf')).softmax(-1)  # [B, H, T, T]
        diag = A.diagonal(dim1=-2, dim2=-1)[..., SKIP:]  # [B, H, T-SKIP]
        local = sum(A.diagonal(offset=-o, dim1=-2, dim2=-1)[..., SKIP - o:] for o in range(1, 9))  # 1..8 tokens back
        out['self_w'].append(diag.mean().item())
        out['self_w_head'].append(diag.mean((0, 2)).tolist())
        out['local_w'].append(local.mean().item())
        if keep_maps:
            out['maps'].append(A[0, :, :N, :N].float().cpu().numpy())
        x = b(x)
        out['rank1'].append(rank1_distance(x))
    return out


def build(variant, ckpt=None, **kw):
    torch.manual_seed(1337)  # train.py seeds the same way, so step-0 weights match the runs exactly
    m = GPT(GPTConfig(variant=variant, **kw)).cuda().eval()
    step = 0
    if ckpt:
        ck = torch.load(ckpt, map_location='cuda')
        m.load_state_dict(ck['model'])
        step = ck['step']
    return m, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/local/scratch/a/liu3170/data/openwebtext')
    ap.add_argument('--runs', default='/local/scratch/a/liu3170/runs/openwebtext')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seq', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    val = np.memmap(os.path.join(args.data, 'val.bin'), dtype=np.uint16, mode='r')
    train = np.memmap(os.path.join(args.data, 'train.bin'), dtype=np.uint16, mode='r')
    counts = np.bincount(train[:500_000_000], minlength=50257).astype(np.float64) + 1.0
    uni = counts / counts.sum()
    uni_ce = float(-np.log(uni[val]).mean())
    print(f'unigram cross-entropy on val: {uni_ce:.4f}', flush=True)
    idx = torch.from_numpy(val[:args.n_seq * T].astype(np.int64)).view(args.n_seq, T).cuda()

    init_settings = [
        ('baseline', 'baseline', {}), ('approach_3', 'approach_3', {}),
        ('three_residual', 'three_residual', {}),
        ('diag_bias 3', 'three_residual', {'diag_bias': 3.0}), ('diag_bias 5', 'three_residual', {'diag_bias': 5.0}),
        ('diag_bias 8', 'three_residual', {'diag_bias': 8.0}),
        ('mimetic 0.8', 'three_residual', {'mimetic': 0.8}), ('mimetic 1.0', 'three_residual', {'mimetic': 1.0}),
        ('mimetic 1.5', 'three_residual', {'mimetic': 1.5}),
    ]
    probe, maps = {}, {}
    for name, v, kw in init_settings:
        m, _ = build(v, **kw)
        r = measure(m, idx, keep_maps=name in ('three_residual', 'diag_bias 8'))
        if r['maps']:
            maps[f'init|{name}'] = np.stack(r.pop('maps'))
        r.pop('maps', None)
        r['params'] = m.num_params()
        probe[name] = r
        print(f'init {name:15s} rank1 L11={r["rank1"][-1]:.3f} self_w L0={r["self_w"][0]:.3f} L11={r["self_w"][-1]:.3f}', flush=True)
    json.dump(probe, open(os.path.join(args.out, 'probe_init.json'), 'w'), indent=1)

    R = args.runs
    trained_models = [
        ('three_residual collapsed', 'three_residual', f'{R}/three_residual/ckpt.pt', {}),
        ('three_residual annealed', 'three_residual', f'{R}/tr_diag8_anneal1000/ckpt.pt', {'diag_bias': 8.0}),
        ('baseline annealed', 'baseline', f'{R}/baseline_diag8_anneal1000/ckpt.pt', {'diag_bias': 8.0}),
        ('approach_3', 'approach_3', f'{R}/approach_3/ckpt.pt', {}),
    ]
    trained = {'unigram_val_ce': uni_ce, 'models': {}}
    for name, v, ck, kw in trained_models:
        m, step = build(v, ck, **kw)
        r = measure(m, idx, keep_maps=True)
        maps[f'trained|{name}'] = np.stack(r.pop('maps'))
        r['step'] = step
        r['diag_bias_now'] = float(m.h[0].attn.diag[0]) if hasattr(m.h[0].attn, 'diag') else None
        r['params'] = m.num_params()
        trained['models'][name] = r
        print(f'{name:26s} step {step} rank1 L11={r["rank1"][-1]:.3f} self_w by layer '
              + ' '.join(f'{s:.2f}' for s in r['self_w']), flush=True)
    json.dump(trained, open(os.path.join(args.out, 'trained.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(args.out, 'attention.npz'), **maps)  # each [12 layers, 12 heads, N, N]
    tokens = val[:N].tolist()
    json.dump({'tokens': tokens}, open(os.path.join(args.out, 'attention_tokens.json'), 'w'))


if __name__ == '__main__':
    main()
