import argparse
import json
import math
import time
from pathlib import Path
import torch
from model import Config, LanguageModel, parameter_counts, forward_flops_per_token


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/minimal")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--train-tokens", type=int, default=100_000_000)
    p.add_argument("--val-tokens", type=int, default=1_000_000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--n-values", type=int, default=1_048_576)
    p.add_argument("--pkm-heads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--d-query", type=int, default=64)
    p.add_argument("--recurrence", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-lr-mult", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default="A,B,C")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def prepare_data(args, tokenizer):
    from datasets import load_dataset
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    seq_tokens = args.seq_len + 1
    needed = {
        "train": max(args.train_tokens, args.steps * args.batch_size * seq_tokens + seq_tokens),
        "validation": max(args.val_tokens, args.eval_batches * args.batch_size * seq_tokens + seq_tokens),
    }
    train_path = root / f"train_{args.seq_len}_{needed['train']}.pt"
    val_path = root / f"validation_{args.seq_len}_{needed['validation']}.pt"
    if train_path.exists() and val_path.exists():
        return train_path, val_path
    stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    iterator = iter(stream)
    buffers = {"train": [], "validation": []}
    sizes = {"train": 0, "validation": 0}
    split = "train"
    while split:
        row = next(iterator)
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        buffers[split].extend(ids)
        sizes[split] += len(ids)
        if sizes[split] >= needed[split]:
            tensor = torch.tensor(buffers[split][:needed[split]], dtype=torch.long)
            path = root / f"{split}_{args.seq_len}_{needed[split]}.pt"
            torch.save(tensor, path)
            print(f"saved {path} ({len(tensor):,} tokens)", flush=True)
            split = "validation" if split == "train" else None
    return root / f"train_{args.seq_len}_{needed['train']}.pt", root / f"validation_{args.seq_len}_{needed['validation']}.pt"


def batches(path, batch_size, seq_len, seed):
    data = torch.load(path, weights_only=True)
    generator = torch.Generator().manual_seed(seed)
    while True:
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
        sample = torch.stack([data[s:s + seq_len + 1] for s in starts])
        yield sample[:, :-1], sample[:, 1:]


@torch.no_grad()
def evaluate(model, loader, batches_count, device):
    model.eval()
    losses = []
    for _ in range(batches_count):
        x, y = next(loader)
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    mean = sum(losses) / len(losses)
    return {"loss": mean, "ppl": math.exp(mean)}


def wide_ff_for_match(base_cfg, memory_params):
    best = base_cfg.ff_mult
    for ff in range(base_cfg.ff_mult, 257):
        cfg = Config(**{**base_cfg.__dict__, "ff_mult": ff, "use_pkm": False})
        model = LanguageModel(cfg)
        if parameter_counts(model)["total"] >= memory_params:
            best = ff
            del model
            break
        del model
    return best


def build_model(name, args):
    cfg = Config(vocab_size=50257, d_model=args.d_model, layers=args.layers, heads=args.heads,
                 ff_mult=args.ff_mult, max_seq_len=args.seq_len, use_pkm=name == "C",
                 pkm_layer=args.layers // 2, n_values=args.n_values, pkm_heads=args.pkm_heads,
                 top_k=args.top_k, d_query=args.d_query, recurrence=args.recurrence)
    if name == "B":
        c = LanguageModel(Config(**{**cfg.__dict__, "use_pkm": True}))
        target = parameter_counts(c)["total"]
        del c
        cfg.ff_mult = wide_ff_for_match(cfg, target)
    return LanguageModel(cfg)


def train(name, args, train_path, val_path):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(name, args).to(device)
    counts = parameter_counts(model)
    value_params = [model.pkm.values] if model.pkm is not None else []
    value_ids = {id(p) for p in value_params}
    main_params = [p for p in model.parameters() if id(p) not in value_ids]
    groups = [{"params": main_params, "lr": args.lr, "weight_decay": 0.1}]
    if value_params:
        groups.append({"params": value_params, "lr": args.lr * args.value_lr_mult, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)
    train_loader = batches(train_path, args.batch_size, args.seq_len, args.seed)
    val_loader = batches(val_path, args.batch_size, args.seq_len, args.seed + 1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"{name}.jsonl"
    used = set()
    eval_history = []
    batch_coverage = []
    batch_entropy = []
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        def record(row):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(name, json.dumps(row), flush=True)
        record({"event": "start", "config": model.cfg.__dict__, **counts,
                "forward_flops_per_token": forward_flops_per_token(model.cfg)})
        for step in range(1, args.steps + 1):
            model.train()
            x, y = next(train_loader)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x.to(device), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if model.pkm is not None:
                flat_indices = model.pkm.last_indices.flatten()
                used.update(flat_indices.cpu().tolist())
                unique, frequency = flat_indices.unique(return_counts=True)
                probabilities = frequency.float() / frequency.sum()
                batch_coverage.append(unique.numel() / args.n_values)
                batch_entropy.append((-(probabilities * probabilities.log()).sum()).item())
            if step % args.log_every == 0 or step == 1:
                metrics = evaluate(model, val_loader, args.eval_batches, device)
                row = {"event": "eval", "step": step, "train_loss": loss.item(), **metrics,
                       "elapsed_sec": time.time() - start}
                if model.pkm is not None:
                    row.update({"gate_mean": model.pkm.last_gate_mean,
                                "observed_unique_cells": len(used),
                                "observed_cell_fraction": len(used) / args.n_values,
                                "batch_unique_cell_fraction": sum(batch_coverage) / len(batch_coverage),
                                "batch_access_entropy_nats": sum(batch_entropy) / len(batch_entropy)})
                    batch_coverage.clear()
                    batch_entropy.clear()
                eval_history.append(row)
                record(row)
        final = evaluate(model, val_loader, args.eval_batches, device)
        record({"event": "final", **final})
        tail = [r["loss"] for r in eval_history if r["step"] > args.steps - 1000]
        record({"event": "last_1000_steps", "validation_loss_mean": sum(tail) / len(tail),
                "eval_points": len(tail)})
        if model.pkm is not None:
            model.disable_pkm = True
            ablated = evaluate(model, val_loader, args.eval_batches, device)
            model.disable_pkm = False
            record({"event": "pkm_ablation", **ablated, "ppl_ratio": ablated["ppl"] / final["ppl"]})
            torch.manual_seed(args.seed + 2)
            model.pkm.random_replace = True
            replaced = evaluate(model, val_loader, args.eval_batches, device)
            model.pkm.random_replace = False
            record({"event": "random_cell_replacement", **replaced,
                    "ppl_ratio": replaced["ppl"] / final["ppl"]})
        torch.save(model.state_dict(), output / f"{name}.pt")
    return final


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_path, val_path = prepare_data(args, tokenizer)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with (Path(args.output) / "args.json").open("w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2)
    for name in args.models.split(","):
        train(name, args, train_path, val_path)


if __name__ == "__main__":
    main()
