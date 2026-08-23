#!/usr/bin/env python3
"""Evaluate one finished run; merge accuracy into its metrics.json.

Missing glue: train_arm() writes loss only, but the placebo gate and
paired tests read accuracy_tail from metrics.json.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluate import concept_frequencies, stratify_eval, evaluate
from src.ontology.mesh import load as load_concepts
from src.ontology.tagger import ConceptTagger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval", default="data/eval/medexqa.jsonl")
    ap.add_argument("--tags", default="results/tags/tags.jsonl")
    ap.add_argument("--concepts", default="results/tags/concepts.json")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    run_dir = Path(a.runs) / f"{a.arm}_{a.budget}_{a.seed}"
    metrics = json.loads((run_dir / "metrics.json").read_text())

    eval_items = [json.loads(l) for l in open(a.eval)]
    if a.limit:
        eval_items = eval_items[: a.limit]
    pool_records = [json.loads(l) for l in open(a.tags)]
    freq = concept_frequencies(pool_records)
    tagger = ConceptTagger(load_concepts(a.concepts))
    strata = stratify_eval(eval_items, tagger, freq)

    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, attn_implementation="sdpa", device_map="cuda")
    model = PeftModel.from_pretrained(model, run_dir / "adapter")
    model.eval()

    metrics.update(evaluate(model, tok, eval_items, strata))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
