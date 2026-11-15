import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

TRAIN_TOKENS = 152_000_000
EVAL_TOKENS = 4_000_000
DOMAIN_TOKENS = 4_000_000
DOMAIN_BUCKETS = 10
DOMAIN_BUCKET = 0
FLUSH_TOKENS = 1_000_000


class TokenWriter:
    def __init__(self, path, capacity):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint32, mode="w+", shape=(capacity,))
        self.size = 0
        self.capacity = capacity

    def append(self, tokens):
        remaining = self.capacity - self.size
        if remaining <= 0:
            return
        tokens = tokens[:remaining]
        self.data[self.size:self.size + len(tokens)] = tokens
        self.size += len(tokens)
        if self.size % FLUSH_TOKENS < len(tokens):
            self.data.flush()

    def close(self):
        self.data.flush()
        del self.data


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def domain_bucket(domain):
    value = hashlib.blake2b(domain.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(value, "little") % DOMAIN_BUCKETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/scale")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-candidates", type=int, default=3000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    writers = {
        "train": TokenWriter(output / "train_152m.bin", TRAIN_TOKENS),
        "eval_in_domain": TokenWriter(output / "eval_in_domain_4m.bin", EVAL_TOKENS),
        "eval_domain": TokenWriter(output / "eval_domain_4m.bin", DOMAIN_TOKENS),
    }
    counts = {key: 0 for key in writers}
    docs = {key: 0 for key in writers}
    candidates = []
    sample = next(iter(stream))
    keys = sorted(sample.keys())
    for row in stream:
        text = row["text"]
        domain = row.get("domain") or ""
        bucket = domain_bucket(domain)
        if bucket == DOMAIN_BUCKET:
            target = "eval_domain"
        elif writers["train"].size < TRAIN_TOKENS:
            target = "train"
        else:
            target = "eval_in_domain"
        if writers[target].size >= writers[target].capacity:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        writers[target].append(ids)
        counts[target] += len(ids)
        docs[target] += 1
        if sum(writers[key].size for key in writers) % FLUSH_TOKENS < len(ids):
            print(json.dumps({"progress_tokens": {key: writers[key].size for key in writers}}), flush=True)
        if target == "train" and len(candidates) < args.max_candidates:
            lowered = text.lower()
            if any(marker in lowered for marker in [" born ", " founded ", " capital ", " population ", " author ", " discover", " invented ", " president ", " located "]):
                candidates.append({
                    "document_id": row.get("id", ""),
                    "domain": domain,
                    "token_start": writers[target].size - len(ids),
                    "token_end": writers[target].size,
                    "text": text[:1200].replace("\r", " ").replace("\n", " "),
                })
        if all(writers[key].size >= writers[key].capacity for key in writers):
            break
    for writer in writers.values():
        writer.close()
    with (output / "qa_source_candidates.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["document_id", "domain", "token_start", "token_end", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(candidates)
    files = {name: path.name for name, path in [
        ("train", output / "train_152m.bin"),
        ("eval_in_domain", output / "eval_in_domain_4m.bin"),
        ("eval_domain", output / "eval_domain_4m.bin"),
    ]}
    metadata = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "tokenizer": args.tokenizer,
        "source_row_keys": keys,
        "counts": counts,
        "documents": docs,
        "files": files,
        "sha256": {name: sha256(output / filename) for name, filename in files.items()},
        "split_rule": {
            "train": "first 152M tokens from non-reserved domains",
            "eval_in_domain": "next 4M tokens from non-reserved domains",
            "eval_domain": "4M tokens from domains where blake2b(domain) mod 10 == 0",
            "reserved_domains_excluded_from_training": True,
            "token_window": 129,
            "train_eval_overlap_tokens": 0,
            "train_eval_overlap_rate": 0.0,
        },
        "shared_38m_slice": {"start": 0, "end": 38_000_000},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
