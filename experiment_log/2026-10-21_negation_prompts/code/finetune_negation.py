"""Fine-tune a pretrained checkpoint to ignore negated distractors ("X is not in D. Therefore, X is in the city of C").

Data: synthetic prompts built from hand-written facts (landmark -> city, country -> capital). The loss covers only
the answer tokens. Every step also takes an OpenWebText batch (plain LM loss) so the model keeps its general skill.
Everything about Paris / France / the Eiffel Tower is excluded from training, so test_prompts.txt stays held out.
A further 25% of the entities are held out to measure whether the skill transfers to unseen facts.

Writes to --out: finetuned.pt (model only), results.json (before/after metrics), log.jsonl.
"""
import argparse
import json
import os
import random

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from gpt2_torch import GPT, GPTConfig

# (subject as it starts a sentence, city). Paris / France facts are deliberately absent.
LANDMARKS = [
    ('The Colosseum', 'Rome'), ('The Brandenburg Gate', 'Berlin'), ('The Reichstag', 'Berlin'),
    ('Big Ben', 'London'), ('The Tower of London', 'London'), ('Buckingham Palace', 'London'),
    ('Wembley Stadium', 'London'), ('The Statue of Liberty', 'New York'), ('The Empire State Building', 'New York'),
    ('Times Square', 'New York'), ('Central Park', 'New York'), ('The Brooklyn Bridge', 'New York'),
    ('The Golden Gate Bridge', 'San Francisco'), ('Alcatraz', 'San Francisco'), ('The Space Needle', 'Seattle'),
    ('The White House', 'Washington'), ('The Lincoln Memorial', 'Washington'), ('The Smithsonian', 'Washington'),
    ('The Willis Tower', 'Chicago'), ('Wrigley Field', 'Chicago'), ('The Hollywood sign', 'Los Angeles'),
    ('The French Quarter', 'New Orleans'), ('The Alamo', 'San Antonio'), ('The Gateway Arch', 'St. Louis'),
    ('The Liberty Bell', 'Philadelphia'), ('Fenway Park', 'Boston'), ('Graceland', 'Memphis'),
    ('The Grand Ole Opry', 'Nashville'), ('The Rock and Roll Hall of Fame', 'Cleveland'),
    ('The Sydney Opera House', 'Sydney'), ('The Kremlin', 'Moscow'), ('Red Square', 'Moscow'),
    ('The Hermitage Museum', 'St. Petersburg'), ('The Acropolis', 'Athens'), ('The Parthenon', 'Athens'),
    ('The Sagrada Familia', 'Barcelona'), ('Camp Nou', 'Barcelona'), ('The Prado Museum', 'Madrid'),
    ('The Alhambra', 'Granada'), ('The Leaning Tower of Pisa', 'Pisa'), ('The Rialto Bridge', 'Venice'),
    ('The Uffizi Gallery', 'Florence'), ('The Forbidden City', 'Beijing'), ('Tiananmen Square', 'Beijing'),
    ('The Bund', 'Shanghai'), ('The Taj Mahal', 'Agra'), ('The Burj Khalifa', 'Dubai'), ('The CN Tower', 'Toronto'),
    ('The Hagia Sophia', 'Istanbul'), ('The Blue Mosque', 'Istanbul'), ('The Rijksmuseum', 'Amsterdam'),
    ('The Little Mermaid statue', 'Copenhagen'), ('Charles Bridge', 'Prague'), ('The Atomium', 'Brussels'),
    ('Schonbrunn Palace', 'Vienna'), ('The Petronas Towers', 'Kuala Lumpur'), ('Old Trafford', 'Manchester'),
    ('Anfield', 'Liverpool'), ('The Western Wall', 'Jerusalem'),
]
CAPITALS = [
    ('Germany', 'Berlin'), ('Italy', 'Rome'), ('Spain', 'Madrid'), ('Portugal', 'Lisbon'),
    ('the United Kingdom', 'London'), ('Ireland', 'Dublin'), ('the Netherlands', 'Amsterdam'),
    ('Belgium', 'Brussels'), ('Austria', 'Vienna'), ('Switzerland', 'Bern'), ('Poland', 'Warsaw'),
    ('the Czech Republic', 'Prague'), ('Hungary', 'Budapest'), ('Greece', 'Athens'), ('Sweden', 'Stockholm'),
    ('Norway', 'Oslo'), ('Denmark', 'Copenhagen'), ('Finland', 'Helsinki'), ('Russia', 'Moscow'),
    ('Turkey', 'Ankara'), ('Egypt', 'Cairo'), ('Japan', 'Tokyo'), ('China', 'Beijing'), ('South Korea', 'Seoul'),
    ('Thailand', 'Bangkok'), ('Vietnam', 'Hanoi'), ('Indonesia', 'Jakarta'), ('the Philippines', 'Manila'),
    ('Australia', 'Canberra'), ('Canada', 'Ottawa'), ('Mexico', 'Mexico City'), ('Argentina', 'Buenos Aires'),
    ('Chile', 'Santiago'), ('Peru', 'Lima'), ('Colombia', 'Bogota'), ('Venezuela', 'Caracas'), ('Cuba', 'Havana'),
    ('Kenya', 'Nairobi'), ('Nigeria', 'Abuja'), ('Ethiopia', 'Addis Ababa'), ('Iran', 'Tehran'), ('Iraq', 'Baghdad'),
    ('Syria', 'Damascus'), ('Lebanon', 'Beirut'), ('Saudi Arabia', 'Riyadh'), ('Afghanistan', 'Kabul'),
    ('Pakistan', 'Islamabad'), ('Bangladesh', 'Dhaka'), ('Romania', 'Bucharest'), ('Bulgaria', 'Sofia'),
    ('Serbia', 'Belgrade'), ('Croatia', 'Zagreb'), ('Iceland', 'Reykjavik'), ('Scotland', 'Edinburgh'),
    ('Wales', 'Cardiff'), ('Morocco', 'Rabat'), ('Algeria', 'Algiers'), ('Tunisia', 'Tunis'), ('Libya', 'Tripoli'),
    ('Sudan', 'Khartoum'), ('Uganda', 'Kampala'), ('Ghana', 'Accra'), ('Senegal', 'Dakar'), ('Zimbabwe', 'Harare'),
    ('Jordan', 'Amman'), ('Qatar', 'Doha'), ('North Korea', 'Pyongyang'), ('Taiwan', 'Taipei'),
    ('Malaysia', 'Kuala Lumpur'), ('Nepal', 'Kathmandu'), ('Ecuador', 'Quito'), ('Uruguay', 'Montevideo'),
    ('Jamaica', 'Kingston'), ('New Zealand', 'Wellington'), ('Estonia', 'Tallinn'), ('Latvia', 'Riga'),
    ('Lithuania', 'Vilnius'), ('Belarus', 'Minsk'), ('Georgia', 'Tbilisi'), ('Armenia', 'Yerevan'),
    ('Azerbaijan', 'Baku'), ('Slovakia', 'Bratislava'), ('Slovenia', 'Ljubljana'), ('Cyprus', 'Nicosia'),
    ('Malta', 'Valletta'), ('Albania', 'Tirana'), ('Somalia', 'Mogadishu'), ('Rwanda', 'Kigali'),
    ('Angola', 'Luanda'), ('Mali', 'Bamako'), ('Oman', 'Muscat'),
]
EXTRA_CITIES = ['Chicago', 'Houston', 'Dallas', 'Miami', 'Denver', 'Detroit', 'Atlanta', 'Phoenix', 'Toronto',
                'Montreal', 'Munich', 'Milan', 'Naples', 'Lyon', 'Hamburg', 'Osaka', 'Mumbai', 'Sydney']
CITIES = sorted({c for _, c in LANDMARKS + CAPITALS} | set(EXTRA_CITIES))


def make_prompt(kind, subj, city, rng, form):
    """form: 'plain' | 'neg' (one negated distractor) | 'rep' (the same negation 2-6 times). Returns (prompt, answer, distractor)."""
    d = rng.choice([c for c in CITIES if c != city])
    if kind == 'landmark':
        q, neg = f'{subj} is in the city of', f'{subj} is not in {d}.'
    else:
        q, neg = f'The capital of {subj} is', f'The capital of {subj} is not {d}.'
        q_then = f'the capital of {subj} is'
    if form == 'plain':
        return q, city, None
    n = 1 if form == 'neg' else rng.randint(2, 6)
    q2 = q if kind == 'landmark' else q_then
    return ' '.join([neg] * n) + f' Therefore, {q2}', city, d


def split_entities(seed=0):
    rng = random.Random(seed)
    out = {}
    for kind, facts in [('landmark', LANDMARKS), ('capital', CAPITALS)]:
        f = facts[:]
        rng.shuffle(f)
        k = len(f) // 4
        out[kind] = {'heldout': f[:k], 'train': f[k:]}
    return out


def eval_set(split, which, seed=1):
    """Fixed list of (form, kind, prompt, answer, distractor) for the given entity split."""
    rng = random.Random(seed)
    items = []
    for kind in ['landmark', 'capital']:
        for subj, city in split[kind][which]:
            for form in ['plain', 'neg', 'rep']:
                p, a, d = make_prompt(kind, subj, city, rng, form)
                items.append((form, kind, p, a, d))
    return items


@torch.no_grad()
def score(m, enc, items):
    """Per form: greedy-correct rate, rate of answering the distractor, mean log-prob of the answer's first token."""
    res = {}
    for form, kind, p, a, d in items:
        ids = torch.tensor([enc.encode(p)], device='cuda')
        n = len(enc.encode(' ' + a))
        x = ids
        for _ in range(n):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = m(x)[0][:, -1]
            x = torch.cat([x, lg.argmax(-1, keepdim=True)], 1)
        out = enc.decode(x[0, ids.shape[1]:].tolist())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lp = m(ids)[0][0, -1].float().log_softmax(-1)
        r = res.setdefault(form, {'n': 0, 'correct': 0, 'distractor': 0, 'logp': 0.0})
        r['n'] += 1
        r['correct'] += out.strip() == a
        r['distractor'] += d is not None and out.strip().startswith(d.split()[0])
        r['logp'] += lp[enc.encode(' ' + a)[0]].item()
    return {f: {'acc': r['correct'] / r['n'], 'distractor_rate': r['distractor'] / r['n'], 'logp': r['logp'] / r['n'],
                'n': r['n']} for f, r in res.items()}


@torch.no_grad()
def test_prompts(m, enc, path):
    prompts = [l.rstrip('\n') for l in open(path) if l.strip() and not l.startswith('#')]
    out = []
    for p in prompts:
        x = torch.tensor([enc.encode(p)], device='cuda')
        n0 = x.shape[1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = m(x)[0][0, -1].float().softmax(-1)
        cand = {c: round(pr[enc.encode(' ' + c)[0]].item(), 4) for c in ['Paris', 'Chicago', 'Berlin', 'London']}
        for _ in range(8):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                x = torch.cat([x, m(x)[0][:, -1:].argmax(-1)], 1)
        out.append({'prompt': p, 'greedy': enc.decode(x[0, n0:].tolist()), 'p': cand})
    return out


@torch.no_grad()
def owt_loss(m, val, n_seq=64, T=1024):
    tot = 0.0
    for i in range(0, n_seq, 8):
        x = torch.from_numpy(val[i * T:(i + 8) * T + 1].astype(np.int64)).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = m(x[:-1].view(8, T), x[1:].view(8, T))
        tot += loss.item()
    return tot / (n_seq // 8)


def evaluate_all(m, enc, split, val, prompts_path):
    m.eval()
    r = {'heldout': score(m, enc, eval_set(split, 'heldout')), 'train_entities': score(m, enc, eval_set(split, 'train')),
         'owt_val_loss': owt_loss(m, val), 'test_prompts': test_prompts(m, enc, prompts_path)}
    m.train()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='pretraining run dir (config.json + ckpt.pt)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='/local/scratch/a/liu3170/data/openwebtext')
    ap.add_argument('--prompts', default='/local/scratch/a/liu3170/gpt-2/test_prompts.txt')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=64, help='synthetic prompts per step')
    ap.add_argument('--owt_batch', type=int, default=8, help='OpenWebText 1024-token sequences per step (0 = none)')
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cfg = json.load(open(os.path.join(args.run, 'config.json')))
    m = GPT(GPTConfig(variant=cfg['variant'], diag_bias=cfg.get('diag_bias', 0.0),
                      diag_learn=cfg.get('diag_learn', False))).cuda()
    m.load_state_dict(torch.load(os.path.join(args.run, 'ckpt.pt'), map_location='cuda')['model'])
    enc = tiktoken.get_encoding('gpt2')
    split = split_entities()
    val = np.memmap(os.path.join(args.data, 'val.bin'), dtype=np.uint16, mode='r')
    train = np.memmap(os.path.join(args.data, 'train.bin'), dtype=np.uint16, mode='r')

    results = {'args': vars(args), 'split': {k: {w: [s for s, _ in v[w]] for w in v} for k, v in split.items()},
               'before': evaluate_all(m, enc, split, val, args.prompts)}
    print('before', json.dumps({k: results['before'][k] for k in ('heldout', 'owt_val_loss')}), flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    log = open(os.path.join(args.out, 'log.jsonl'), 'w')
    m.train()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / 20)  # short warmup, then constant
        for g in opt.param_groups:
            g['lr'] = lr
        # synthetic batch: right-padded, loss on answer tokens only
        seqs, tgts = [], []
        for _ in range(args.batch):
            kind = rng.choice(['landmark', 'capital'])
            subj, city = rng.choice(split[kind]['train'])
            form = rng.choices(['plain', 'neg', 'rep'], weights=[1, 2, 1])[0]
            p, a, _ = make_prompt(kind, subj, city, rng, form)
            pi, ai = enc.encode(p), enc.encode(' ' + a + '.')
            seqs.append(pi + ai)
            tgts.append([-100] * (len(pi) - 1) + ai + [-100])
        L = max(map(len, seqs))
        x = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device='cuda')
        y = torch.tensor([t + [-100] * (L - len(t)) for t in tgts], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = m(x)[0]
            loss_ft = F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1), ignore_index=-100)
            loss = loss_ft
            if args.owt_batch:
                ix = np.random.randint(0, len(train) - 1025, args.owt_batch)
                b = torch.from_numpy(np.stack([train[i:i + 1025] for i in ix]).astype(np.int64)).cuda()
                _, loss_lm = m(b[:, :-1].contiguous(), b[:, 1:].contiguous())
                loss = loss + loss_lm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
        opt.step()
        rec = {'step': step, 'loss_ft': loss_ft.item(), 'loss_lm': loss_lm.item() if args.owt_batch else None,
               'grad_norm': gn}
        log.write(json.dumps(rec) + '\n')
        if step % 50 == 0:
            print(rec, flush=True)
            log.flush()

    results['after'] = evaluate_all(m, enc, split, val, args.prompts)
    print('after', json.dumps({k: results['after'][k] for k in ('heldout', 'owt_val_loss')}), flush=True)
    for t in results['after']['test_prompts']:
        print(repr(t['greedy']), t['p'], flush=True)
    json.dump(results, open(os.path.join(args.out, 'results.json'), 'w'), indent=1)
    torch.save({'model': m.state_dict(), 'step': args.steps}, os.path.join(args.out, 'finetuned.pt'))


if __name__ == '__main__':
    main()
