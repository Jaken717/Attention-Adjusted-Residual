"""Validation loss of the pretrained OpenAI GPT-2 124M on the same val.bin, as a reference point."""
import argparse
import json

import torch

from gpt2_torch import load_openai_gpt2
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', type=int, default=20_000_000)
    args = ap.parse_args()
    model = load_openai_gpt2('baseline').cuda()
    vl = evaluate(model, f'{args.data}/val.bin', 32, 1024, args.tokens, 'cuda')
    rec = {'model': 'openai-gpt2-124M (pretrained)', 'val_loss': vl, 'eval_tokens': args.tokens}
    print(json.dumps(rec))
    with open(args.out, 'w') as f:
        json.dump(rec, f, indent=2)


if __name__ == '__main__':
    main()
