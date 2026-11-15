import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

NON_DOMAIN_CAPACITY = 44_000_000
DOMAIN_CAPACITY = 2_000_000
FLUSH_TOKENS = 1_000_000
DOMAIN_BUCKETS = 10
DOMAIN_BUCKET = 0


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
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-dir", default="data/scale/shards")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    stream = stream.shard(num_shards=args.num_shards, index=args.shard_index)
    non_domain = TokenWriter(output / f"part{args.shard_index}_non_domain.bin", NON_DOMAIN_CAPACITY)
    domain = TokenWriter(output / f"part{args.shard_index}_domain.bin", DOMAIN_CAPACITY)
    docs = {"non_domain": 0, "domain": 0}
    candidates = []
    for row in stream:
        domain_name = row.get("domain") or row.get("url") or ""
        target_name = "domain" if domain_bucket(domain_name) == DOMAIN_BUCKET else "non_domain"
        writer = domain if target_name == "domain" else non_domain
        if writer.size >= writer.capacity:
            if non_domain.size >= non_domain.capacity and domain.size >= domain.capacity:
                break
            continue
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        writer.append(ids)
        docs[target_name] += 1
        if target_name == "non_domain" and len(candidates) < 1000:
            lowered = row["text"].lower()
            if any(marker in lowered for marker in [" born ", " founded ", " capital ", " population ", " author ", " discover", " invented ", " president ", " located "]):
                candidates.append({
                    "shard": args.shard_index,
                    "document_id": row.get("id", ""),
                    "domain": domain_name,
                    "part_token_start": writer.size - len(ids),
                    "part_token_end": writer.size,
                    "text": row["text"][:1200].replace("\r", " ").replace("\n", " "),
                })
        if (non_domain.size + domain.size) % FLUSH_TOKENS < len(ids):
            print(json.dumps({"shard": args.shard_index, "non_domain": non_domain.size, "domain": domain.size}), flush=True)
    non_domain.close()
    domain.close()
    with (output / f"part{args.shard_index}_qa_candidates.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["shard", "document_id", "domain", "part_token_start", "part_token_end", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(candidates)
    metadata = {
        "shard": args.shard_index,
        "num_shards": args.num_shards,
        "counts": {"non_domain": non_domain.size, "domain": domain.size},
        "documents": docs,
        "files": {
            "non_domain": non_domain.path.name,
            "domain": domain.path.name,
        },
        "sha256": {
            "non_domain": sha256(non_domain.path),
            "domain": sha256(domain.path),
        },
    }
    (output / f"part{args.shard_index}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
