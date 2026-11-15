from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from pkm import ProductKeyMemory


@dataclass
class ScaleConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    layers: int = 36
    heads: int = 16
    ff_mult: int = 4
    max_seq_len: int = 1024
    n_values: int = 0
    pkm_layer: int = 18
    pkm_heads: int = 4
    top_k: int = 8
    d_query: int = 64
    pkm_dv: int = 512
    dropout: float = 0.0


class ScaleBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.d_model // cfg.heads
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.attn_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        b, t, d = x.shape
        n = self.norm1(x)
        qkv = self.qkv(n).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.attn_out(a)
        return x + self.ff(self.norm2(x))


class SparseAdamW:
    def __init__(self, values, lr, betas=(0.9, 0.999), eps=1e-8):
        self.values = values
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.step_count = 0
        self.exp_avg = torch.zeros_like(values)
        self.exp_avg_sq = torch.zeros_like(values)

    @torch.no_grad()
    def step(self, indices, gradients):
        indices = indices.reshape(-1)
        gradients = gradients.reshape(-1, gradients.shape[-1]).float().cpu()
        unique, inverse = indices.unique(return_inverse=True)
        summed = torch.zeros(len(unique), gradients.shape[1])
        summed.index_add_(0, inverse, gradients)
        counts = torch.bincount(inverse, minlength=len(unique)).unsqueeze(1)
        gradients = summed / counts
        self.step_count += 1
        beta1, beta2 = self.betas
        m = self.exp_avg.index_select(0, unique)
        v = self.exp_avg_sq.index_select(0, unique)
        m.mul_(beta1).add_(gradients, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(gradients, gradients, value=1 - beta2)
        m_hat = m / (1 - beta1 ** self.step_count)
        v_hat = v / (1 - beta2 ** self.step_count)
        self.values.index_add_(0, unique, -self.lr * m_hat / (v_hat.sqrt() + self.eps))
        self.exp_avg.index_copy_(0, unique, m)
        self.exp_avg_sq.index_copy_(0, unique, v)


class ScaleLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.positions = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(ScaleBlock(cfg) for _ in range(cfg.layers))
        self.pkm = ProductKeyMemory(cfg.d_model, cfg.n_values, cfg.pkm_heads, cfg.top_k,
                                    cfg.d_query, cfg.pkm_dv, sparse_offload=True) if cfg.n_values else None
        if self.pkm is not None:
            self.pkm.values.requires_grad_(False)
        self.value_grad_batches = []
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tokens.weight
        self.apply(self.init_weights)
        self.disable_pkm = False

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def to_device(self, device):
        self.tokens.to(device)
        self.positions.to(device)
        self.blocks.to(device)
        self.norm.to(device)
        self.lm_head.to(device)
        if self.pkm is not None:
            self.pkm.query.to(device)
            self.pkm.query_norm.to(device)
            self.pkm.keys1 = nn.Parameter(self.pkm.keys1.to(device))
            self.pkm.keys2 = nn.Parameter(self.pkm.keys2.to(device))
            self.pkm.output.to(device)
            self.pkm.gate.to(device)
            self.pkm.values = nn.Parameter(self.pkm.values.cpu())
            self.pkm.values.requires_grad_(False)
        return self

    def collect_value_gradients(self):
        if self.pkm is None or self.pkm.last_selected is None or self.pkm.last_selected.grad is None:
            return
        self.value_grad_batches.append((
            self.pkm.last_selected_indices.reshape(-1).cpu(),
            self.pkm.last_selected.grad.detach().cpu(),
        ))

    def apply_value_gradients(self, optimizer):
        if not self.value_grad_batches:
            return
        indices = torch.cat([item[0] for item in self.value_grad_batches])
        gradients = torch.cat([item[1] for item in self.value_grad_batches])
        optimizer.step(indices, gradients)
        self.value_grad_batches.clear()

    def hidden(self, tokens):
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.tokens(tokens) + self.positions(pos)[None]
        for i, block in enumerate(self.blocks, 1):
            x = checkpoint(block, x, use_reentrant=False) if torch.is_grad_enabled() else block(x)
            if self.pkm is not None and i == self.cfg.pkm_layer and not self.disable_pkm:
                x = self.pkm(x)
        return self.norm(x)

    def loss(self, hidden, targets, chunk_size=4096):
        total = torch.zeros((), device=hidden.device)
        count = targets.numel()
        for start in range(0, hidden.shape[0] * hidden.shape[1], chunk_size):
            rows = hidden.reshape(-1, hidden.shape[-1])[start:start + chunk_size]
            labels = targets.reshape(-1)[start:start + chunk_size]
            total = total + F.cross_entropy(self.lm_head(rows), labels, reduction="sum")
        return total / count

    def forward(self, tokens, targets=None):
        hidden = self.hidden(tokens)
        return hidden, self.loss(hidden, targets) if targets is not None else None

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=0.0):
        self.eval()
        for _ in range(max_new_tokens):
            context = tokens[:, -self.cfg.max_seq_len:]
            logits = self.lm_head(self.hidden(context))[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(logits.softmax(dim=-1) / temperature, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    values = model.pkm.values.numel() if model.pkm is not None else 0
    transformer = sum(p.numel() for p in model.blocks.parameters())
    return {
        "total": total,
        "active": total - values,
        "value_parameters": values,
        "transformer_parameters": transformer,
        "value_bytes_fp32": values * 4,
    }
