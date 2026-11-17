"""从 run 目录的 checkpoint 生成样本（贪心，默认 200 token）。

用法:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/venv_train/bin/python gen_sample.py \
        runs/dense_control/dense152_lr0.0003_seed1 "Question: When was Marco Polo born?\nAnswer:"

注意：prompt 里的 \n 要写成真实换行（引号内直接回车）或 $'...'。
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scale_model import ScaleConfig, ScaleLanguageModel  # noqa: E402


def main():
    run = Path(sys.argv[1])
    prompts = sys.argv[2:] or ["Question: When was Marco Polo born?\nAnswer:"]
    args = json.loads((run / "args.json").read_text(encoding="utf-8"))
    cfg = ScaleConfig(
        d_model=args["d_model"], layers=args["layers"], heads=args["heads"],
        ff_mult=args["ff_mult"], max_seq_len=args["seq_len"], n_values=args["n_values"],
        pkm_layer=args["pkm_layer"], pkm_heads=args["pkm_heads"], top_k=args["top_k"],
        d_query=args["d_query"], pkm_dv=args["pkm_dv"],
    )
    device = torch.device("cuda")
    model = ScaleLanguageModel(cfg)
    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to_device(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated = model.generate(ids, max_new_tokens=200)
        print("=" * 24, repr(prompt[:64]))
        print(tokenizer.decode(generated[0], skip_special_tokens=True))
        print()


if __name__ == "__main__":
    main()
