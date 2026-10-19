"""Diagnostics for all attention-residual variants (baseline, approach_1..5), at init and at the last checkpoint.

Writes to --out:
  variants_probe.json : per-layer stats per model (init and trained), KL to unigram, attention-term magnitudes
  alpha_sweep.json    : approach_5 rank-collapse probe at init over a grid of alpha
  matched_evals.json  : every trained checkpoint (and the pretrained GPT-2, the unigram) on identical val subsets
  heatmaps.npz        : head-mean attention, first 48 positions, layers 0/5/11, trained models
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from gpt2_torch import GPT, GPTConfig, load_openai_gpt2
from train import evaluate

SKIP = 64
MODELS = [  # name, variant, alpha, run dir
    ('baseline', 'baseline', 0.9, 'baseline'),
    ('approach_1', 'approach_1', 0.9, 'approach_1'),
    ('approach_2', 'approach_2', 0.9, 'approach_2'),
    ('approach_3', 'approach_3', 0.9, 'approach_3'),
    ('approach_4', 'approach_4', 0.9, 'approach_4_full'),
    ('approach_5', 'approach_5', 0.9, 'approach_5_a0.9'),
]


def stream_stats(x):
    xs = x[:, SKIP:].float()
    xn = F.normalize(xs, dim=-1)
    cos = (xn @ xn.transpose(1, 2)).mean().item()
    centered = xs - xs.mean(1, keepdim=True)
    rank1 = (centered.norm(dim=(1, 2)) / xs.norm(dim=(1, 2))).mean().item()
    return cos, rank1, xs.norm(dim=-1).mean().item()


@torch.no_grad()
def probe(model, idx, want_heat=False):
    cfg = model.cfg
    T, D, H = idx.shape[1], cfg.n_embd, cfg.n_head
    mask = torch.ones(T, T, device=idx.device).tril().bool()
    unif = torch.log(torch.arange(1, T + 1, device=idx.device).float())
    x = model.wte(idx) + model.wpe(torch.arange(T, device=idx.device))
    cos, r1, nrm = stream_stats(x)
    layers, heat = [{'layer': 'emb', 'cos': cos, 'rank1_res': r1, 'norm': nrm}], {}

    def merge(t):
        return t.transpose(1, 2).reshape(t.shape[0], T, D)

    for l, b in enumerate(model.h):
        at = b.attn
        q, k, v = at.c_attn(b.ln_1(x)).split(D, 2)
        q, k, v = at.split_heads(q), at.split_heads(k), at.split_heads(v)
        A = (q @ k.transpose(-1, -2) / (D // H) ** 0.5).masked_fill(~mask, float('-inf')).softmax(-1)
        ent = -(A * A.clamp_min(1e-12).log()).sum(-1)
        rec = {'layer': l,
               'ent_ratio': (ent[..., SKIP:] / unif[SKIP:]).mean().item(),
               'self_w': A.diagonal(dim1=-2, dim2=-1)[..., SKIP:].mean().item()}
        yv, yx = merge(A @ v)[:, SKIP:], merge(A @ at.split_heads(x))[:, SKIP:]  # Concat[A V], Concat[A X_h]
        xs = x[:, SKIP:]
        Wo = at.c_proj.weight.T  # [D, D], x @ Wo
        n = lambda t: t.norm(dim=-1).mean().item()
        rec['n_attnV'] = n(yv @ Wo)  # standard attention term (no bias)
        rec['n_x'] = n(xs)
        rec['n_mixX'] = n(yx)  # Concat[A X_h]
        rec['blur'] = (yx - xs).norm(dim=-1).mean().item() / rec['n_x']  # how far mixing moves X, relative to |X|
        if cfg.variant == 'approach_3':
            rec['n_mixX_Wo'] = n(yx @ Wo)  # the new X_h term inside attention
        if want_heat and l in (0, 5, 11):
            heat[l] = A[0, :, :48, :48].mean(0).float().cpu().numpy()
        x = b(x)  # propagate with the real module
        cos, r1, nrm = stream_stats(x)
        rec.update({'cos': cos, 'rank1_res': r1, 'norm': nrm})
        layers.append(rec)
    logits = (model.ln_f(x) @ model.wte.weight.T)[..., :50257].float()
    return layers, heat, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'

    train = np.memmap(os.path.join(args.data, 'train.bin'), dtype=np.uint16, mode='r')
    val = np.memmap(os.path.join(args.data, 'val.bin'), dtype=np.uint16, mode='r')
    counts = np.bincount(train[:500_000_000], minlength=50257).astype(np.float64) + 1.0
    uni = counts / counts.sum()
    uni_t = torch.tensor(uni, device=dev, dtype=torch.float32)
    val_path = os.path.join(args.data, 'val.bin')

    def uni_ce(n_tokens):
        n = n_tokens // 32768 * 32768
        return float(-np.log(uni[val[1:n + 1]]).mean())

    matched = {'val4M': {'unigram': uni_ce(4_194_304)}, 'val20M': {'unigram': uni_ce(20_000_000)}, 'steps': {}}
    ref = load_openai_gpt2().to(dev)
    matched['val4M']['openai_gpt2_124M'] = evaluate(ref, val_path, 32, 1024, 4_194_304, dev)
    matched['val20M']['openai_gpt2_124M'] = evaluate(ref, val_path, 32, 1024, 20_000_000, dev)
    del ref

    idx = torch.from_numpy(val[:8 * 1024].astype(np.int64)).view(8, 1024).to(dev)
    probes, heats = {}, {}
    for name, variant, alpha, run in MODELS:
        ck_path = os.path.join(args.runs, run, 'ckpt.pt')
        states = [('init', None)]
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
            states.append(('trained', ck))
        for tag, ck in states:
            torch.manual_seed(1337)
            m = GPT(GPTConfig(variant=variant, alpha=alpha)).to(dev).eval()
            if ck is not None:
                m.load_state_dict(ck['model'])
            layers, heat, logits = probe(m, idx, want_heat=ck is not None)
            logp = logits[:, SKIP:].log_softmax(-1)
            p = logp.exp()
            entry = {'layers': layers,
                     'kl_to_unigram': (p * (logp - uni_t.log())).sum(-1).mean().item(),
                     'pred_std_across_pos': p.std(1).mean().item()}
            if ck is not None:
                entry['step'] = ck['step']
                matched['steps'][name] = ck['step']
                matched['val4M'][name] = evaluate(m, val_path, 32, 1024, 4_194_304, dev)
                matched['val20M'][name] = evaluate(m, val_path, 32, 1024, 20_000_000, dev)
                for l, a in heat.items():
                    heats[f'{name}|L{l}'] = a
            probes[f'{name}@{tag}'] = entry
            print(f'{name}@{tag}: L11 cos={layers[-1]["cos"]:.3f} rank1={layers[-1]["rank1_res"]:.3f} '
                  f'KL_uni={entry["kl_to_unigram"]:.4f}', flush=True)

    sweep = []
    for a in [0.0, 0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0]:
        torch.manual_seed(1337)
        m = GPT(GPTConfig(variant='approach_5', alpha=a)).to(dev).eval()
        layers, _, _ = probe(m, idx)
        sweep.append({'alpha': a, 'rank1_L11': layers[-1]['rank1_res'], 'cos_L11': layers[-1]['cos'],
                      'rank1_by_layer': [r['rank1_res'] for r in layers]})
    json.dump(probes, open(os.path.join(args.out, 'variants_probe.json'), 'w'), indent=1)
    json.dump(sweep, open(os.path.join(args.out, 'alpha_sweep.json'), 'w'), indent=1)
    json.dump(matched, open(os.path.join(args.out, 'matched_evals.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(args.out, 'heatmaps.npz'), **heats)
    print(json.dumps(matched, indent=1))


if __name__ == '__main__':
    main()
