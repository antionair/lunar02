import csv
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

TARGET_TOKENS = 38_000_000
DOMAIN_BUCKETS = 10
DOMAIN_BUCKET = 0
OUTPUT = Path("data/scale/qa_source_candidates.tsv")


def bucket(value):
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % DOMAIN_BUCKETS


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    rows = []
    tokens = 0
    docs = 0
    for row in stream:
        domain = row.get("domain") or row.get("url") or ""
        if bucket(domain) == DOMAIN_BUCKET:
            continue
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        start = tokens
        tokens += len(ids)
        docs += 1
        lowered = row["text"].lower()
        markers = [" born ", " died ", " founded ", " established ", " published ", " capital ", " population ", " author ", " written by ", " discover", " invented ", " created by ", " developed by ", " designed by ", " built by ", " composed by ", " directed by ", " painted by ", " president ", " located ", " headquartered ", " official language ", " currency ", " nickname ", " motto ", " students ", " employees ", " members ", " covers an area "]
        if any(marker in lowered for marker in markers):
            rows.append({
                "document_id": row.get("id", ""),
                "domain": domain,
                "token_start": start,
                "token_end": tokens,
                "text": row["text"][:6000].replace("\r", " ").replace("\n", " "),
            })
        if tokens >= TARGET_TOKENS:
            break
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["document_id", "domain", "token_start", "token_end", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"tokens": tokens, "documents": docs, "candidates": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
