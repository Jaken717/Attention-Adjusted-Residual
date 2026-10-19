"""Diagnostics for the attention-residual collapse.

Per layer and per checkpoint (and at initialization), on validation sequences:
  ent_ratio : attention entropy / entropy of uniform attention over the causal prefix (1 = uniform)
  self_w    : mean attention weight a token puts on itself, A[i, i]
  cos       : mean cosine similarity between residual vectors at different positions (1 = all identical)
  rank1_res : ||X - 1 mean(X)||_F / ||X||_F, distance of the residual stream from rank one (0 = collapsed)
  norm      : mean L2 norm of the residual stream
Also the unigram cross-entropy of the val set and how close each model's predictions are to the unigram.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from gpt2_torch import GPT, GPTConfig, VARIANTS

SKIP = 64  # ignore the first positions, where the causal prefix is short


@torch.no_grad()
def probe(model, idx):
    T = idx.shape[1]
    D = model.cfg.n_embd
    mask = torch.ones(T, T, device=idx.device).tril().bool()
    unif = torch.log(torch.arange(1, T + 1, device=idx.device).float())
    x = model.wte(idx) + model.wpe(torch.arange(T, device=idx.device))
    layers, heat = [], {}

    def stream_stats(x):
        xs = x[:, SKIP:].float()
        xn = F.normalize(xs, dim=-1)
        cos = (xn @ xn.transpose(1, 2)).mean().item()
        centered = xs - xs.mean(1, keepdim=True)
        rank1 = (centered.norm(dim=(1, 2)) / xs.norm(dim=(1, 2))).mean().item()
        return cos, rank1, xs.norm(dim=-1).mean().item()

    cos0, r0, n0 = stream_stats(x)
    layers.append({'layer': 'emb', 'cos': cos0, 'rank1_res': r0, 'norm': n0})
    for l, b in enumerate(model.h):
        at = b.attn
        q, k, _ = at.c_attn(b.ln_1(x)).split(D, 2)
        q, k = at.split_heads(q), at.split_heads(k)
        A = (q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)).masked_fill(~mask, float('-inf')).softmax(-1)
        ent = -(A * A.clamp_min(1e-12).log()).sum(-1)
        ent_ratio = (ent[..., SKIP:] / unif[SKIP:]).mean().item()
        self_w = A.diagonal(dim1=-2, dim2=-1)[..., SKIP:].mean().item()
        heat[l] = A[0, :, :48, :48].float().cpu().numpy()  # all heads, first 48 positions of sequence 0
        x = b(x)
        cos, rank1, nrm = stream_stats(x)
        layers.append({'layer': l, 'ent_ratio': ent_ratio, 'self_w': self_w, 'cos': cos, 'rank1_res': rank1,
                       'norm': nrm})
    logits = (model.ln_f(x) @ model.wte.weight.T)[..., :50257].float()
    return layers, heat, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seq', type=int, default=8)
    ap.add_argument('--unigram_tokens', type=int, default=500_000_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'

    train = np.memmap(os.path.join(args.data, 'train.bin'), dtype=np.uint16, mode='r')
    val = np.memmap(os.path.join(args.data, 'val.bin'), dtype=np.uint16, mode='r')
    counts = np.bincount(train[:args.unigram_tokens], minlength=50257).astype(np.float64) + 1.0
    uni = counts / counts.sum()
    uni_ce = float(-np.log(uni[val]).mean())
    uni_entropy = float(-(uni * np.log(uni)).sum())
    print(f'unigram cross-entropy on val: {uni_ce:.4f}  unigram entropy: {uni_entropy:.4f}', flush=True)
    uni_t = torch.tensor(uni, device=dev, dtype=torch.float32)

    idx = torch.from_numpy(val[:args.n_seq * 1024].astype(np.int64)).view(args.n_seq, 1024).to(dev)
    results = {'unigram_val_ce': uni_ce, 'unigram_entropy': uni_entropy, 'models': {}}
    heats = {}
    mean_preds = {'unigram': uni.astype(np.float32)}
    for v in VARIANTS:
        ck_path = os.path.join(args.runs, v, 'ckpt.pt')
        states = [('init', None)]
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
            states.append((f'step{ck["step"]}', ck['model']))
        for tag, sd in states:
            torch.manual_seed(1337)  # same init as train.py
            m = GPT(GPTConfig(variant=v)).to(dev).eval()
            if sd is not None:
                m.load_state_dict(sd)
            layers, heat, logits = probe(m, idx)
            logp = logits[:, SKIP:].log_softmax(-1)
            p = logp.exp()
            kl_uni = (p * (logp - uni_t.log())).sum(-1).mean().item()  # KL(model || unigram), per position
            pred_std = p.std(1).mean().item()  # spread of predictions across positions
            tgt = idx[:, SKIP + 1:]
            ce = F.nll_loss(logp[:, :-1].reshape(-1, logp.shape[-1]), tgt.reshape(-1)).item()
            key = f'{v}@{tag}'
            results['models'][key] = {'layers': layers, 'kl_to_unigram': kl_uni, 'pred_std_across_pos': pred_std,
                                      'probe_ce': ce}
            heats[key] = heat
            mean_preds[key] = p.mean((0, 1)).cpu().numpy()  # average predicted next-token distribution
            print(f'{key:24s} probe_ce={ce:.3f} KL(model||unigram)={kl_uni:.4f} pred_std={pred_std:.2e} '
                  f'final cos={layers[-1]["cos"]:.3f} rank1_res={layers[-1]["rank1_res"]:.3f}', flush=True)

    with open(os.path.join(args.out, 'collapse_probe.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out, 'attention_heatmaps.npz'),
                        **{f'{k}|L{l}': a for k, h in heats.items() for l, a in h.items()})
    np.savez_compressed(os.path.join(args.out, 'mean_predictions.npz'), **mean_preds)


if __name__ == '__main__':
    main()
