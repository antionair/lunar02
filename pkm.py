import math
import torch
from torch import nn


class ProductKeyMemory(nn.Module):
    def __init__(self, d_model, n_values=1_048_576, heads=4, top_k=8, d_query=64, value_dim=None, sparse_offload=False):
        super().__init__()
        if int(math.sqrt(n_values)) ** 2 != n_values:
            raise ValueError("n_values must be a perfect square")
        if d_query % 2:
            raise ValueError("d_query must be even")
        self.n_values = n_values
        self.side = int(math.sqrt(n_values))
        self.heads = heads
        self.top_k = top_k
        self.query = nn.Linear(d_model, heads * d_query)
        self.query_norm = nn.LayerNorm(heads * d_query)
        self.keys1 = nn.Parameter(torch.empty(heads, self.side, d_query // 2))
        self.keys2 = nn.Parameter(torch.empty(heads, self.side, d_query // 2))
        self.value_dim = value_dim or d_model
        self.values = nn.Parameter(torch.empty(n_values, self.value_dim))
        self.output = nn.Linear(self.value_dim, d_model)
        self.gate = nn.Linear(d_model, d_model)
        nn.init.normal_(self.keys1, std=d_query ** -0.5)
        nn.init.normal_(self.keys2, std=d_query ** -0.5)
        nn.init.normal_(self.values, std=self.value_dim ** -0.5)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.last_indices = None
        self.random_replace = False
        self.sparse_offload = sparse_offload
        self.last_selected = None
        self.last_selected_indices = None

    def forward(self, h):
        shape = h.shape
        h = h.reshape(-1, shape[-1])
        q = self.query_norm(self.query(h))
        q = q.view(h.shape[0], self.heads, 2, -1)
        q1, q2 = q[:, :, 0], q[:, :, 1]
        scores1 = torch.einsum("bhd,hnd->bhn", q1, self.keys1)
        scores2 = torch.einsum("bhd,hnd->bhn", q2, self.keys2)
        s1, i1 = scores1.topk(self.top_k, dim=-1)
        s2, i2 = scores2.topk(self.top_k, dim=-1)
        candidate_scores = s1.unsqueeze(-1) + s2.unsqueeze(-2)
        candidate_indices = i1.unsqueeze(-1) * self.side + i2.unsqueeze(-2)
        flat_scores = candidate_scores.flatten(2)
        scores, candidate = flat_scores.topk(self.top_k, dim=-1)
        indices = candidate_indices.flatten(2).gather(-1, candidate)
        weights = scores.softmax(dim=-1)
        self.last_indices = indices.detach()
        if self.random_replace:
            indices = torch.randint_like(indices, self.n_values)
        if self.sparse_offload:
            cpu_indices = indices.detach().cpu()
            selected = self.values[cpu_indices].to(indices.device)
            if torch.is_grad_enabled():
                selected.requires_grad_(True)
            self.last_selected = selected
            self.last_selected_indices = cpu_indices
        elif self.values.device != indices.device:
            selected = self.values[indices.detach().cpu()].to(indices.device)
        else:
            selected = self.values[indices]
        memory = torch.einsum("bhk,bhkd->bd", weights, selected) / self.heads
        memory = memory.reshape(*shape[:-1], self.value_dim)
        gate = torch.sigmoid(self.gate(h.reshape(shape)))
        result = h.reshape(shape) + gate * self.output(memory)
        self.last_gate_mean = gate.detach().mean().item()
        return result
