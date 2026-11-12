from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from pkm import ProductKeyMemory


@dataclass
class Config:
    vocab_size: int = 50257
    d_model: int = 512
    layers: int = 8
    heads: int = 8
    ff_mult: int = 4
    max_seq_len: int = 512
    use_pkm: bool = False
    pkm_layer: int = 4
    n_values: int = 1_048_576
    pkm_heads: int = 4
    top_k: int = 8
    d_query: int = 64
    dropout: float = 0.0
    recurrence: int = 1


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, mask):
        n = self.norm1(x)
        a = self.attn(n, n, n, attn_mask=mask, need_weights=False)[0]
        x = x + a
        return x + self.ff(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.positions = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.pkm = ProductKeyMemory(cfg.d_model, cfg.n_values, cfg.pkm_heads, cfg.top_k, cfg.d_query) if cfg.use_pkm else None
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

    def forward(self, tokens, targets=None):
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.tokens(tokens) + self.positions(pos)[None]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tokens.device), diagonal=1)
        for i, block in enumerate(self.blocks, 1):
            x = block(x, mask)
            if i == self.cfg.pkm_layer:
                if self.pkm is not None and not self.disable_pkm:
                    x = self.pkm(x)
                for _ in range(self.cfg.recurrence):
                    for recurrent_block in self.blocks[i:]:
                        x = recurrent_block(x, mask)
                break
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()) if targets is not None else None
        return logits, loss


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    memory = model.pkm.values.numel() if model.pkm is not None else 0
    return {"total": total, "active": total - memory, "memory_values": memory}


def forward_flops_per_token(cfg):
    executed_blocks = cfg.pkm_layer + (cfg.layers - cfg.pkm_layer) * cfg.recurrence
    d = cfg.d_model
    block_flops = (8 + 4 * cfg.ff_mult) * d * d + 2 * cfg.max_seq_len * d
    result = executed_blocks * block_flops + 2 * cfg.vocab_size * d
    if cfg.use_pkm:
        side = int(math.sqrt(cfg.n_values))
        result += 2 * d * (cfg.pkm_heads * cfg.d_query + 2 * d)
        result += 2 * cfg.pkm_heads * side * cfg.d_query
        result += 2 * cfg.pkm_heads * cfg.top_k * d
    return result
