import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SHARDS = ROOT / "data/scale/shards"
OUTPUT = ROOT / "data/scale"
TRAIN_TOKENS = 152_000_000
EVAL_TOKENS = 4_000_000
DOMAIN_TOKENS = 4_000_000


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actual_size(path):
    data = np.memmap(path, dtype=np.uint32, mode="r")
    nonzero = np.nonzero(data)[0]
    if len(nonzero) == 0:
        return 0
    return int(nonzero[-1]) + 1


def concatenate(parts, output, limit):
    data = np.memmap(output, dtype=np.uint32, mode="w+", shape=(limit,))
    cursor = 0
    offsets = []
    for part in parts:
        source = np.memmap(part, dtype=np.uint32, mode="r")
        count = min(len(source), limit - cursor)
        data[cursor:cursor + count] = source[:count]
        offsets.append({"file": part.name, "start": cursor, "count": count})
        cursor += count
        if cursor == limit:
            break
    if cursor != limit:
        raise RuntimeError(f"only merged {cursor} tokens into {output}")
    data.flush()
    del data
    return offsets


def main():
    non_domain_parts = [SHARDS / f"part{i}_non_domain.bin" for i in range(4)]
    domain_parts = [SHARDS / f"domain{i}.bin" for i in range(4)]
    non_domain_counts = [actual_size(path) for path in non_domain_parts]
    domain_counts = [actual_size(path) for path in domain_parts]
    if sum(non_domain_counts) < TRAIN_TOKENS + EVAL_TOKENS:
        raise RuntimeError(f"non-domain total too small: {sum(non_domain_counts)}")
    if sum(domain_counts) < DOMAIN_TOKENS:
        raise RuntimeError(f"domain total too small: {sum(domain_counts)}")
    merged_non_domain = OUTPUT / "non_domain_156m.bin"
    non_domain_offsets = concatenate(non_domain_parts, merged_non_domain, TRAIN_TOKENS + EVAL_TOKENS)
    data = np.memmap(merged_non_domain, dtype=np.uint32, mode="r")
    train = np.memmap(OUTPUT / "train_152m.bin", dtype=np.uint32, mode="w+", shape=(TRAIN_TOKENS,))
    train[:] = data[:TRAIN_TOKENS]
    train.flush()
    eval_in_domain = np.memmap(OUTPUT / "eval_in_domain_4m.bin", dtype=np.uint32, mode="w+", shape=(EVAL_TOKENS,))
    eval_in_domain[:] = data[TRAIN_TOKENS:TRAIN_TOKENS + EVAL_TOKENS]
    eval_in_domain.flush()
    del data, train, eval_in_domain
    merged_non_domain.unlink()
    domain_offsets = concatenate(domain_parts, OUTPUT / "eval_domain_4m.bin", DOMAIN_TOKENS)
    files = {"train": "train_152m.bin", "eval_in_domain": "eval_in_domain_4m.bin", "eval_domain": "eval_domain_4m.bin"}
    payload = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "tokenizer": "gpt2",
        "counts": {"train": TRAIN_TOKENS, "eval_in_domain": EVAL_TOKENS, "eval_domain": DOMAIN_TOKENS},
        "files": files,
        "sha256": {name: sha256(OUTPUT / filename) for name, filename in files.items()},
        "shard_counts": {"non_domain": non_domain_counts, "domain": domain_counts},
        "merge_offsets": {"non_domain": non_domain_offsets, "domain": domain_offsets},
        "split_rule": {
            "train": "first 152M tokens from non-reserved URL/domain hash shards",
            "eval_in_domain": "next 4M tokens from non-reserved URL/domain hash shards",
            "eval_domain": "first 4M tokens where blake2b(domain-or-url) mod 10 == 0",
            "reserved_domains_excluded_from_training": True,
            "qa_source_limit": "documents starting inside first 38M training tokens",
            "train_eval_overlap_tokens": 0,
            "train_eval_overlap_rate": 0.0,
        },
        "shared_38m_slice": {"start": 0, "end": 38_000_000},
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
