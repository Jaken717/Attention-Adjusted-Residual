"""Train a GPT-2 124M variant from scratch on tokenized FineWeb-Edu, with gradient accumulation.

Single GPU: python train.py ...   Multi-GPU (DDP): torchrun --nproc_per_node=4 train.py ...
All variants see the same batches in the same order (sequential read from train.bin). Each optimizer
step consumes the same batch_tokens contiguous tokens regardless of the number of GPUs; the GPUs split
that step's micro-batches between them.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2_torch import GPT, GPTConfig, VARIANTS


class Loader:
    """Micro-batch `micro` of optimizer step `step` on this rank, addressed directly into train.bin.

    Step s covers tokens [s*batch_tokens, (s+1)*batch_tokens); rank r takes micro-batches
    r*grad_accum .. (r+1)*grad_accum-1 of it. With one GPU this is a plain sequential read.
    """
    def __init__(self, path, B, T, grad_accum, rank=0):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.B, self.T, self.grad_accum, self.rank = B, T, grad_accum, rank

    def get(self, step, micro, batch_tokens):
        n = self.B * self.T
        pos = (step * batch_tokens + (self.rank * self.grad_accum + micro) * n) % (len(self.data) - n - 1)
        buf = torch.from_numpy(self.data[pos:pos + n + 1].astype(np.int64))
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)


@torch.no_grad()
def evaluate(model, path, B, T, max_tokens, device, rank=0, world=1):
    """Mean loss over the first max_tokens of path; batches are split across ranks and summed."""
    model.eval()
    data = np.memmap(path, dtype=np.uint16, mode='r')
    n = min(max_tokens, len(data) - 1) // (B * T) * (B * T)
    acc = torch.zeros(2, device=device)  # [sum of batch losses, number of batches]
    for k, i in enumerate(range(0, n, B * T)):
        if k % world != rank:
            continue
        buf = torch.from_numpy(data[i:i + B * T + 1].astype(np.int64)).to(device)
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        acc += torch.stack([loss.detach().float(), torch.ones((), device=device)])
    if world > 1:
        dist.all_reduce(acc)
    model.train()
    return (acc[0] / acc[1]).item()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    ratio = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * ratio)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=VARIANTS, required=True)
    ap.add_argument('--alpha', type=float, default=0.9, help='approach_5: weight of the unmixed skip X')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', type=float, default=2.5e9)
    ap.add_argument('--batch_tokens', type=int, default=524288)
    ap.add_argument('--micro_batch', type=int, default=32)
    ap.add_argument('--seq_len', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--min_lr_frac', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=250)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--eval_tokens', type=int, default=4_194_304)
    ap.add_argument('--final_eval_tokens', type=int, default=20_000_000)
    ap.add_argument('--ckpt_every', type=int, default=1000)
    ap.add_argument('--max_steps', type=int, default=None, help='stop early (benchmarking)')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--no_compile', action='store_true')
    args = ap.parse_args()

    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
    else:
        rank, world, local_rank = 0, 1, 0
    master = rank == 0
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    if master:
        os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)  # same seed on every rank -> identical initial weights
    torch.set_float32_matmul_precision('high')

    B, T = args.micro_batch, args.seq_len
    assert args.batch_tokens % (B * T * world) == 0
    grad_accum = args.batch_tokens // (B * T * world)  # micro-batches per rank per step
    args.steps = int(args.tokens // args.batch_tokens)

    model = GPT(GPTConfig(variant=args.variant, alpha=args.alpha)).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    start = 0
    ckpt_path = os.path.join(args.out, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start = ck['step']
        if master:
            print(f'resumed from step {start}', flush=True)

    cmodel = model if args.no_compile else torch.compile(model)  # used directly for eval (no collectives)
    tmodel = DDP(cmodel, device_ids=[local_rank]) if ddp else cmodel  # used for training
    train = Loader(os.path.join(args.data, 'train.bin'), B, T, grad_accum, rank)
    val_path = os.path.join(args.data, 'val.bin')
    args.world_size = world
    log = None
    if master:
        log = open(os.path.join(args.out, 'log.jsonl'), 'a')
        print(f'variant={args.variant} params={model.num_params():,} steps={args.steps} '
              f'gpus={world} grad_accum_per_gpu={grad_accum}', flush=True)
        with open(os.path.join(args.out, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    end = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    t0 = time.time()
    for step in range(start, end):
        if step % args.eval_every == 0:
            vl = evaluate(cmodel, val_path, B, T, args.eval_tokens, device, rank, world)
            rec = {'step': step, 'tokens': step * args.batch_tokens, 'val_loss': vl}
            if master:
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + '\n'); log.flush()

        lr = lr_at(step, args)
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device)
        for micro in range(grad_accum):
            x, y = train.get(step, micro, args.batch_tokens)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if ddp:
                tmodel.require_backward_grad_sync = micro == grad_accum - 1  # all-reduce once per step
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = tmodel(x, y)
            (loss / grad_accum).backward()  # DDP averages over ranks -> mean over all micro-batches
            loss_acc += loss.detach()
        if ddp:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        t0 = time.time()
        train_loss = (loss_acc / grad_accum).item()
        rec = {'step': step, 'train_loss': train_loss, 'lr': lr, 'grad_norm': norm.item(),
               'dt': dt, 'tok_per_s': args.batch_tokens / dt}
        if master:
            log.write(json.dumps(rec) + '\n')
            if step % 10 == 0:
                log.flush()
                print(json.dumps(rec), flush=True)
        if not math.isfinite(train_loss):  # identical on all ranks after the all-reduce
            if master:
                print('non-finite loss, stopping', flush=True)
            break
        if master and ((step + 1) % args.ckpt_every == 0 or step + 1 == args.steps):
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step + 1,
                        'config': vars(args)}, ckpt_path)
    else:
        if end == args.steps:
            vl = evaluate(cmodel, val_path, B, T, args.final_eval_tokens, device, rank, world)
            rec = {'step': end, 'tokens': end * args.batch_tokens, 'val_loss': vl, 'final': True}
            if master:
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + '\n')
    if master:
        log.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
