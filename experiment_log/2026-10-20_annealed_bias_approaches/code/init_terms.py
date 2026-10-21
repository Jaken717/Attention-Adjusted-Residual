"""Size of each attention-sublayer term at initialization, for approach_1 and approach_4 (with and without the bias).

Norms are Frobenius norms over 4 validation sequences, positions >= 64. Writes init_terms.json to --out.
"""
import argparse
import json
import os

import numpy as np
import torch

from three_residual_analysis import build, T

D = 768


@torch.no_grad()
def terms(m, idx):
    x = m.wte(idx) + m.wpe(torch.arange(T, device=idx.device))
    B = idx.shape[0]
    rows = []
    for blk in m.h:
        at = blk.attn
        q, k, v = [at.split_heads(t) for t in at.c_attn(blk.ln_1(x)).split(D, 2)]
        mix = lambda z: at.attend(q, k, z).transpose(1, 2).reshape(B, T, D)  # Concat_h[A_h z_h]
        ax, av = mix(at.split_heads(x)), mix(v)
        n = lambda t: t[:, 64:].norm().item()
        w = at.c_proj.weight.T
        rows.append({'X': n(x), 'AX': n(ax), 'AV': n(av), 'AX_WO': n(ax @ w), 'AV_WO': n(av @ w)})
        x = blk(x)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/local/scratch/a/liu3170/data/openwebtext')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    val = np.memmap(os.path.join(args.data, 'val.bin'), dtype=np.uint16, mode='r')
    idx = torch.from_numpy(val[:4 * T].astype(np.int64)).view(4, T).cuda()
    res = {}
    for v in ['approach_1', 'approach_2', 'approach_4']:
        for b in [0.0, 8.0]:
            m, _ = build(v, diag_bias=b)
            res[f'{v}|b{b:g}'] = terms(m, idx)
            r = res[f'{v}|b{b:g}']
            print(v, f'b={b:g}', ' | '.join(f"L{l} " + ' '.join(f'{k}={r[l][k]:.0f}' for k in r[l]) for l in (0, 11)))
    json.dump(res, open(os.path.join(args.out, 'init_terms.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
