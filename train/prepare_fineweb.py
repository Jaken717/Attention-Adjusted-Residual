"""Download FineWeb-Edu or OpenWebText parquet shards and tokenize with the GPT-2 BPE into uint16 .bin files.

Each document is prefixed with <|endoftext|>. Writes train.bin and val.bin to --out.
OpenWebText has 80 shards (~110M tokens each); --shards picks that many, spread evenly over the 80.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

enc = tiktoken.get_encoding('gpt2')
EOT = enc._special_tokens['<|endoftext|>']


def tokenize(text):
    return np.array([EOT] + enc.encode_ordinary(text), dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', choices=['fineweb_edu', 'openwebtext'], default='fineweb_edu')
    ap.add_argument('--delete_raw', action='store_true', help='delete each parquet shard once tokenized')
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--val_tokens', type=int, default=20_000_000)
    ap.add_argument('--workers', type=int, default=64)
    args = ap.parse_args()

    raw = os.path.join(args.out, 'raw')
    if args.dataset == 'fineweb_edu':
        repo, files = 'HuggingFaceFW/fineweb-edu', [f'sample/10BT/{i:03d}_00000.parquet' for i in range(args.shards)]
    else:
        idx = np.linspace(0, 79, args.shards).round().astype(int)
        repo, files = 'Skylion007/openwebtext', [f'plain_text/train-{i:05d}-of-00080.parquet' for i in idx]

    all_path = os.path.join(args.out, 'all.bin')
    total = 0
    with open(all_path, 'wb') as f, Pool(args.workers) as pool:
        for name in files:
            p = hf_hub_download(repo, name, repo_type='dataset', local_dir=raw)
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                texts = pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
                for toks in pool.imap(tokenize, texts, chunksize=64):
                    f.write(toks.tobytes())
                    total += len(toks)
            print(f'{p}: {total:,} tokens so far', flush=True)
            if args.delete_raw:
                os.remove(p)

    # val = last val_tokens tokens, train = everything before it
    data = np.memmap(all_path, dtype=np.uint16, mode='r')
    split = len(data) - args.val_tokens
    data[split:].tofile(os.path.join(args.out, 'val.bin'))
    del data
    with open(all_path, 'r+b') as f:
        f.truncate(split * 2)
    os.replace(all_path, os.path.join(args.out, 'train.bin'))
    print(f'train: {split:,} tokens, val: {args.val_tokens:,} tokens')


if __name__ == '__main__':
    main()
