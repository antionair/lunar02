import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

CAPACITY = 1_000_000
DOMAIN_BUCKETS = 10
DOMAIN_BUCKET = 0


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket(value):
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % DOMAIN_BUCKETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-dir", default="data/scale/shards")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"domain{args.shard_index}.bin"
    data = np.memmap(path, dtype=np.uint32, mode="w+", shape=(CAPACITY,))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).shard(
        num_shards=args.num_shards, index=args.shard_index)
    size = 0
    docs = 0
    for row in stream:
        domain = row.get("domain") or row.get("url") or ""
        if bucket(domain) != DOMAIN_BUCKET:
            continue
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        ids = ids[:CAPACITY - size]
        if not ids:
            break
        data[size:size + len(ids)] = ids
        size += len(ids)
        docs += 1
        if size % 100000 < len(ids):
            print(json.dumps({"shard": args.shard_index, "domain": size}), flush=True)
        if size >= CAPACITY:
            break
    data.flush()
    del data
    if size != CAPACITY:
        raise RuntimeError(f"shard {args.shard_index} only produced {size} domain tokens")
    metadata = {"shard": args.shard_index, "count": size, "documents": docs,
                "file": path.name, "sha256": sha256(path),
                "domain_rule": "blake2b(domain-or-url) mod 10 == 0"}
    (output / f"domain{args.shard_index}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
