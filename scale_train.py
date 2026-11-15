import argparse
import json
import math
from pathlib import Path
import re
import resource
import time
import unicodedata
import numpy as np
import torch
from scale_model import ScaleConfig, ScaleLanguageModel, SparseAdamW, parameter_counts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", default="data/scale")
    p.add_argument("--train-tokens", type=int, required=True)
    p.add_argument("--n-values", type=int, default=0)
    p.add_argument("--pkm-dv", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-lr-mult", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1000)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--pkm-heads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--d-query", type=int, default=64)
    p.add_argument("--pkm-layer", type=int, default=18)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--qa-file", default="")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_tokens(path):
    return np.memmap(path, dtype=np.uint32, mode="r")


def next_batch(data, cursor, micro_batch, seq_len, device):
    width = micro_batch * (seq_len + 1)
    if cursor + width > len(data):
        cursor = 0
    chunk = torch.from_numpy(data[cursor:cursor + width].astype(np.int64)).view(micro_batch, seq_len + 1)
    return chunk[:, :-1].to(device, non_blocking=True), chunk[:, 1:].to(device, non_blocking=True), cursor + width


@torch.no_grad()
def evaluate(model, data, batches, micro_batch, seq_len, device):
    model.eval()
    cursor = 0
    losses = []
    for _ in range(batches):
        x, y, cursor = next_batch(data, cursor, micro_batch, seq_len, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def entropy(flat_indices):
    unique, counts = flat_indices.unique(return_counts=True)
    probabilities = counts.float() / counts.sum()
    return unique.numel(), (-(probabilities * probabilities.log()).sum()).item()


def normalize_answer(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def token_f1(prediction, reference):
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = {}
    for token in expected:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in predicted:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction, row):
    references = row.get("accepted_aliases") or [row["answer"]]
    normalized_prediction = normalize_answer(prediction)
    exact = max(float(normalized_prediction == normalize_answer(reference)) for reference in references)
    f1 = max(token_f1(prediction, reference) for reference in references)
    return exact, f1


@torch.no_grad()
def evaluate_qa(model, qa_file, device):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    rows = [json.loads(line) for line in Path(qa_file).read_text(encoding="utf-8").splitlines()]
    exact = []
    f1 = []
    predictions = []
    for row in rows:
        prompt = f"Question: {row['question']}\nAnswer:"
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated = model.generate(tokens, max_new_tokens=16)[0, tokens.shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        answer_exact, answer_f1 = score_answer(prediction, row)
        exact.append(answer_exact)
        f1.append(answer_f1)
        predictions.append({"id": row["id"], "prediction": prediction})
    return {"count": len(rows), "exact_match": sum(exact) / len(exact),
            "token_f1": sum(f1) / len(f1), "predictions": predictions}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    train_path = Path(args.data_dir) / "train_152m.bin"
    eval_path = Path(args.data_dir) / "eval_in_domain_4m.bin"
    domain_path = Path(args.data_dir) / "eval_domain_4m.bin"
    train_data = load_tokens(train_path)
    eval_data = load_tokens(eval_path)
    domain_data = load_tokens(domain_path)
    cfg = ScaleConfig(
        d_model=args.d_model, layers=args.layers, heads=args.heads, ff_mult=args.ff_mult,
        max_seq_len=args.seq_len, n_values=args.n_values, pkm_layer=args.pkm_layer,
        pkm_heads=args.pkm_heads, top_k=args.top_k, d_query=args.d_query, pkm_dv=args.pkm_dv,
    )
    model = ScaleLanguageModel(cfg).to_device(device)
    counts = parameter_counts(model)
    value_params = [model.pkm.values] if model.pkm is not None else []
    value_ids = {id(p) for p in value_params}
    base_params = [p for p in model.parameters() if id(p) not in value_ids]
    optimizer = torch.optim.AdamW(base_params, lr=args.lr, weight_decay=0.1)
    value_optimizer = SparseAdamW(model.pkm.values, lr=args.lr * args.value_lr_mult) if model.pkm is not None else None
    effective_tokens = args.micro_batch * args.seq_len * args.grad_accum
    steps = args.train_tokens // effective_tokens
    used = np.zeros(args.n_values, dtype=np.bool_) if args.n_values else None
    cursor = 0
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        def record(row):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        record({"event": "start", "config": cfg.__dict__, "seed": args.seed,
                "main_lr": args.lr, "value_lr": args.lr * args.value_lr_mult,
                "value_lr_rule": "value_lr = main_lr * value_lr_mult",
                "train_tokens": args.train_tokens, "steps": steps,
                "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
                "effective_batch_tokens": effective_tokens, **counts})
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_coverages = []
            batch_entropies = []
            train_loss = 0.0
            for _ in range(args.grad_accum):
                x, y, cursor = next_batch(train_data, cursor, args.micro_batch, args.seq_len, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                (loss / args.grad_accum).backward()
                model.collect_value_gradients()
                train_loss += loss.item() / args.grad_accum
                if model.pkm is not None:
                    flat = model.pkm.last_indices.flatten()
                    unique, value = entropy(flat)
                    used[flat.cpu().numpy()] = True
                    batch_coverages.append(unique / args.n_values)
                    batch_entropies.append(value)
            torch.nn.utils.clip_grad_norm_(base_params, 1.0)
            if value_optimizer is not None:
                model.apply_value_gradients(value_optimizer)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == steps:
                metrics = evaluate(model, eval_data, args.eval_batches, args.micro_batch, args.seq_len, device)
                row = {"event": "eval", "step": step, "train_loss": train_loss,
                       "val_loss": metrics, "val_ppl": math.exp(metrics),
                       "elapsed_sec": time.time() - started,
                       "tokens_per_sec": cursor / (time.time() - started)}
                if model.pkm is not None:
                    row.update({"gate_mean": model.pkm.last_gate_mean,
                                "cumulative_cell_usage": float(used.mean()),
                                "batch_unique_cell_fraction": sum(batch_coverages) / len(batch_coverages),
                                "batch_access_entropy_nats": sum(batch_entropies) / len(batch_entropies)})
                record(row)
        val = evaluate(model, eval_data, 32, args.micro_batch, args.seq_len, device)
        domain = evaluate(model, domain_data, 32, args.micro_batch, args.seq_len, device)
        elapsed = time.time() - started
        record({"event": "final", "val_loss": val, "val_ppl": math.exp(val),
                "domain_loss": domain, "domain_ppl": math.exp(domain),
                "elapsed_sec": elapsed, "tokens_per_sec": args.train_tokens / elapsed,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                "cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
        if args.qa_file:
            qa = evaluate_qa(model, args.qa_file, device)
            predictions = qa.pop("predictions")
            record({"event": "fact_qa", **qa})
            (output / "qa_predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8")
        (output / "args.json").write_text(json.dumps(args.__dict__, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
