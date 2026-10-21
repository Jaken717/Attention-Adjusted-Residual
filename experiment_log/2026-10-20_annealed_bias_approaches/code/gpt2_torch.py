"""PyTorch port of src/model.py with the attention-residual variants.

variant:
  'baseline'   : standard GPT-2.  x = x + Concat_h[A_h V_h] W_O + b_O
  'approach_1' : Y = Concat_h[A_h (V_h + X_h)];          x = Y W_O + b_O + Y
  'approach_2' : Y = LN(Concat_h[A_h (V_h + X_h)]);      x = Y W_O + b_O + Y   (LN without affine)
  'approach_3' : Y = Concat_h[A_h (V_h + X_h)];          x = x + Y W_O + b_O   (identity skip kept)
  'approach_4' : x = Concat_h[A_h V_h] W_O + b_O + Concat_h[A_h X_h]           (mixed X_h replaces the skip)
  'approach_5' : x = Concat_h[A_h V_h] W_O + b_O + alpha X + (1 - alpha) Concat_h[A_h X_h]
  'three_residual': Z_h = A_h X_h + X_h;  Y = Concat_h[Z_h W_vh + Z_h];  x = Y W_O + b_O + Y
                    (W_vh is a per-head D/H x D/H matrix; c_attn makes only Q and K)
  Pretraining aids (the forward formula is unchanged); diag_bias also works for baseline and approach_1..5:
    diag_bias b : add b to each head's self logit (A[i, i]), so A starts close to I. Learnable per head with
                  diag_learn, otherwise a buffer that train.py can anneal to 0 (then the final model is plain three_residual).
    mimetic s   : init W_Q = W_K = s * (random orthogonal), so each query matches its own key best at init.
X_h is head h's D/H slice of the raw (pre-ln_1) residual stream. The MLP residual is standard in all variants.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ('baseline', 'approach_1', 'approach_2', 'approach_3', 'approach_4', 'approach_5', 'three_residual')


@dataclass
class GPTConfig:
    n_vocab: int = 50304  # 50257 padded to a multiple of 64
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    variant: str = 'baseline'
    alpha: float = 0.9  # approach_5 only: weight of the unmixed skip X
    diag_bias: float = 0.0  # three_residual only: initial bias on the self logit
    diag_learn: bool = False  # three_residual only: learn diag_bias per head (else a buffer, annealable)
    mimetic: float = 0.0  # three_residual only: W_Q = W_K = mimetic * orthogonal at init (0 = off)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        three = cfg.variant == 'three_residual'
        self.c_attn = nn.Linear(cfg.n_embd, (2 if three else 3) * cfg.n_embd)
        if three:
            d = cfg.n_embd // cfg.n_head
            self.w_v = nn.Parameter(torch.randn(cfg.n_head, d, d) * 0.02)  # W_vh, one per head
        if cfg.diag_learn or cfg.diag_bias != 0:
            diag = torch.full((cfg.n_head,), float(cfg.diag_bias))
            if cfg.diag_learn:
                self.diag = nn.Parameter(diag)
            else:
                self.register_buffer('diag', diag)
            neg = torch.zeros(cfg.n_ctx, cfg.n_ctx).masked_fill(~torch.ones(cfg.n_ctx, cfg.n_ctx).tril().bool(), float('-inf'))
            self.register_buffer('causal_neg', neg, persistent=False)  # 0 on and below the diagonal, -inf above
            self.register_buffer('eye', torch.eye(cfg.n_ctx), persistent=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.RESID_SCALE_INIT = True

    def split_heads(self, t):
        B, T, D = t.shape
        return t.view(B, T, self.cfg.n_head, D // self.cfg.n_head).transpose(1, 2)  # [B, H, T, D/H]

    def attend(self, q, k, v):
        """Causal softmax(q k^T / sqrt(d) [+ diag_bias * I]) v."""
        if hasattr(self, 'diag'):
            T = q.shape[-2]
            bias = self.causal_neg[:T, :T] + self.diag.view(-1, 1, 1) * self.eye[:T, :T]  # [H, T, T]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def forward(self, x_ln, x_raw):
        B, T, D = x_ln.shape
        if self.cfg.variant == 'three_residual':
            q, k = map(self.split_heads, self.c_attn(x_ln).split(D, dim=2))
            xh = self.split_heads(x_raw)  # X_h
            z = self.attend(q, k, xh) + xh  # Z_h = A_h X_h + X_h
            z = torch.einsum('bhtd,hde->bhte', z, self.w_v) + z  # Z_h W_vh + Z_h
            y = z.transpose(1, 2).contiguous().view(B, T, D)  # Y = Concat_h
            return self.c_proj(y) + y  # Y W_O + b_O + Y
        q, k, v = self.c_attn(x_ln).split(D, dim=2)
        q, k, v = map(self.split_heads, (q, k, v))
        if self.cfg.variant in ('approach_4', 'approach_5'):
            # one attention pass over [V_h | X_h] gives A_h V_h and A_h X_h with the same A_h
            if hasattr(self, 'diag'):
                # two calls: under torch.compile, the masked call on the wide [V_h | X_h] gives NaN gradients
                y, yx = self.attend(q, k, v), self.attend(q, k, self.split_heads(x_raw))
            else:
                vx = torch.cat([v, self.split_heads(x_raw)], dim=-1)  # [B, H, T, 2D/H]
                y, yx = F.scaled_dot_product_attention(q, k, vx, is_causal=True).split(D // self.cfg.n_head, dim=-1)
            yx = yx.transpose(1, 2).contiguous().view(B, T, D)  # Concat_h[A_h X_h]
            y = y.transpose(1, 2).contiguous().view(B, T, D)  # Concat_h[A_h V_h]
            if self.cfg.variant == 'approach_5':
                return self.c_proj(y) + self.cfg.alpha * x_raw + (1 - self.cfg.alpha) * yx
            return self.c_proj(y) + yx
        if self.cfg.variant != 'baseline':
            v = v + self.split_heads(x_raw)  # V_h + X_h
        y = self.attend(q, k, v)  # A_h (V_h [+ X_h]), scale 1/sqrt(D/H)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # Concat_h
        if self.cfg.variant == 'approach_2':
            y = F.layer_norm(y, (D,), eps=1e-5)  # Y <- LN(Y), no affine
        out = self.c_proj(y)  # Y W_O + b_O
        if self.cfg.variant in ('approach_1', 'approach_2'):
            out = out + y  # + Y
        return out


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.RESID_SCALE_INIT = True

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.mlp = MLP(cfg)

    def forward(self, x):
        a = self.attn(self.ln_1(x), x)
        x = x + a if self.cfg.variant in ('baseline', 'approach_3') else a  # approach_1/2: output replaces the skip path
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.variant in VARIANTS, cfg.variant
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.n_vocab, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.apply(self._init_weights)
        if cfg.mimetic:
            assert cfg.variant == 'three_residual'
            D = cfg.n_embd
            with torch.no_grad():
                for b in self.h:
                    w = nn.init.orthogonal_(torch.empty(D, D)) * cfg.mimetic
                    b.attn.c_attn.weight[:D] = w  # W_Q
                    b.attn.c_attn.weight[D:2 * D] = w  # W_K = W_Q

    @torch.no_grad()
    def set_diag(self, value):
        for b in self.h:
            b.attn.diag.fill_(value)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, 'RESID_SCALE_INIT', False):
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        logits = self.ln_f(x) @ self.wte.weight.T  # tied output embedding
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def load_openai_gpt2(variant='baseline'):
    """Load the pretrained OpenAI GPT-2 124M weights (HF safetensors) into this module."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download('openai-community/gpt2', 'model.safetensors'))
    model = GPT(GPTConfig(n_vocab=50257, variant=variant))
    own = model.state_dict()
    new = {}
    for k, v in sd.items():
        k = k.removeprefix('transformer.')
        if k.endswith('.attn.bias') or k.endswith('.attn.masked_bias'):
            continue
        if k.endswith('.weight') and any(s in k for s in ('c_attn', 'c_proj', 'c_fc')):
            v = v.T  # HF Conv1D stores [in, out]; nn.Linear wants [out, in]
        new[k] = v
    missing = set(own) - set(new)
    assert not missing, missing
    model.load_state_dict(new, strict=True)
    return model
