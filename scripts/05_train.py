#!/usr/bin/env python3
"""Train one arm at one seed. Knows nothing about selection."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.train import train_arm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--pool", default="data/pool/pool.jsonl")
    ap.add_argument("--selections", default="results/selections")
    ap.add_argument("--out", default="results/runs")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg["arm"] = a.arm
    sel = json.loads(Path(a.selections, f"{a.arm}_{a.budget}_{a.seed}.json").read_text())
    corpus = [json.loads(l) for l in open(a.pool)]

    out = Path(a.out) / f"{a.arm}_{a.budget}_{a.seed}"
    m = train_arm(corpus, sel["ids"], cfg, a.seed, out)
    print(json.dumps(m, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
