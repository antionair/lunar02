import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
train = torch.load(ROOT / "data/train_128_30000000.pt", weights_only=True)
validation = torch.load(ROOT / "data/validation_128_200000.pt", weights_only=True)
sequence_window = 129
max_end = len(train) - 129 + sequence_window
checks = {
    "train_tokens": len(train),
    "validation_tokens": len(validation),
    "max_sampled_training_end_exclusive": max_end,
    "validation_start_token_index": len(train),
    "training_token_overlap": max(0, max_end - len(train)),
    "training_validation_overlap_rate": max(0, max_end - len(train)) / len(validation),
    "validation_or_neighbor_entered_training": max_end > len(train),
    "sequence_window": sequence_window,
}
output = ROOT / "runs/ffn_lr_probe/data_overlap.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2))
