"""Run the prompts in test_prompts.txt through trained checkpoints (and OpenAI GPT-2 124M for reference).

For each model and prompt, prints the greedy continuation and the next-token probability of a few candidate
answers, plus each candidate's rank among all 50257 tokens.
"""
import argparse
import json
import os

import tiktoken
import torch

from gpt2_torch import GPT, GPTConfig, load_openai_gpt2

RUNS = '/local/scratch/a/liu3170/runs/openwebtext'
MODELS = ['baseline_diag8_anneal1000', 'approach_3', 'tr_diag8_anneal1000', 'approach_2_diag8_anneal1000',
          'approach_1_diag8_anneal1000', 'approach_4_diag8_anneal1000']
CANDIDATES = [' Paris', ' Chicago', ' Berlin', ' London', ' Rome']


def load_run(name):
    cfg = json.load(open(os.path.join(RUNS, name, 'config.json')))
    m = GPT(GPTConfig(variant=cfg['variant'], diag_bias=cfg.get('diag_bias', 0.0), diag_learn=cfg.get('diag_learn', False),
                      mimetic=cfg.get('mimetic', 0.0)))
    ck = torch.load(os.path.join(RUNS, name, 'ckpt.pt'), map_location='cpu')
    m.load_state_dict(ck['model'])
    return m, ck['step']


@torch.no_grad()
def ask(m, enc, prompt, n_new):
    ids = torch.tensor([enc.encode(prompt)], device='cuda')
    probs = m(ids)[0][0, -1].float().softmax(-1)
    cand = {}
    for c in CANDIDATES:
        t = enc.encode(c)[0]
        cand[c.strip()] = (probs[t].item(), int((probs > probs[t]).sum()) + 1)
    for _ in range(n_new):
        ids = torch.cat([ids, m(ids)[0][:, -1:].argmax(-1)], 1)
    return enc.decode(ids[0, -n_new:].tolist()), cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', default='test_prompts.txt')
    ap.add_argument('--n_new', type=int, default=8)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    prompts = [l.rstrip('\n') for l in open(args.prompts) if l.strip() and not l.startswith('#')]
    enc = tiktoken.get_encoding('gpt2')
    results = {}
    names = ['openai_gpt2_124M'] + MODELS
    for name in names:
        if name == 'openai_gpt2_124M':
            m, step = load_openai_gpt2(), 'pretrained'
        else:
            m, step = load_run(name)
        m = m.cuda().eval()
        print(f'\n=== {name} (step {step}) ===', flush=True)
        results[name] = []
        for i, p in enumerate(prompts):
            cont, cand = ask(m, enc, p, args.n_new)
            results[name].append({'prompt': p, 'greedy': cont, 'candidates': cand})
            top = '  '.join(f'{k} {v[0]:.3f} (#{v[1]})' for k, v in cand.items())
            print(f'[P{i + 1}] -> {cont!r}\n      {top}', flush=True)
        del m
        torch.cuda.empty_cache()
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
