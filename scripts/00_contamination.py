#!/usr/bin/env python3
"""
GATE 0: contamination / headroom check.

If base Qwen already scores high on the eval set, there is no room for
fine-tuning to differentiate arms, and every arm will converge. Run this
BEFORE building anything else.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluate import score_multiple_choice

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval", default="data/eval/medexqa.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.eval)][: a.limit]
    n_opt = len(items[0]["options"])
    chance = 1.0 / n_opt

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa",
        device_map="auto").eval()

    correct = sum(
        int(score_multiple_choice(model, tok, it["question"], it["options"])
            == it["answer_idx"]) for it in items)
    acc = correct / len(items)

    print(f"zero-shot accuracy : {acc:.3f}")
    print(f"chance             : {chance:.3f}")
    print(f"headroom           : {1.0 - acc:.3f}")

    if acc > 0.60:
        print("\n*** WARNING: high zero-shot accuracy. Likely contamination or")
        print("*** low difficulty. Ceiling effects will prevent arms from")
        print("*** separating. Consider MedXpertQA (10-option) instead.")
        return 1
    if acc < chance * 1.2:
        print("\n*** WARNING: at chance. Model may be too small for this task.")
        return 1
    print("\nGATE PASSED -> adequate headroom")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
