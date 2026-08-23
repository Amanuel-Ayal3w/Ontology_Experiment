#!/usr/bin/env python3
"""Run one selector, emit an ID list. Times the selection (Proof 2)."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selection.registry import get_selector
from src.ontology.mesh import load as load_concepts
from src.cost import measure, save as save_cost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--budget", type=int, required=True, help="token budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tags", default="results/tags/tags.jsonl")
    ap.add_argument("--pool", default="data/pool/pool.jsonl")
    ap.add_argument("--concepts", default="results/tags/concepts.json")
    ap.add_argument("--out", default="results/selections")
    a = ap.parse_args()

    # Selections are line indices into the pool file. Hash it so any run
    # record can prove which pool the ids index into.
    pool_sha256 = hashlib.sha256(Path(a.pool).read_bytes()).hexdigest()

    records = [json.loads(l) for l in open(a.tags)]
    corpus = [json.loads(l) for l in open(a.pool)]
    # approximate tokens as chars/4 -- replace with real tokenizer counts if preferred
    lengths = [max(r["n_chars"] // 4, 1) for r in records]

    kw = {}
    if a.arm in ("ontology", "placebo"):
        concepts = load_concepts(a.concepts)
        kw["depths"] = {ui: c.depth for ui, c in concepts.items()}
    if a.arm == "d4":
        kw["texts"] = [f"{r.get('instruction','')} {r.get('input','')}" for r in corpus]

    costs = {}
    fn = get_selector(a.arm)
    with measure(f"select_{a.arm}", costs):
        ids = fn(records, lengths, a.budget, seed=a.seed, **kw)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arm}_{a.budget}_{a.seed}"
    (out / f"{tag}.json").write_text(json.dumps({
        "arm": a.arm, "budget_tokens": a.budget, "seed": a.seed,
        "pool_sha256": pool_sha256,
        "n_selected": len(ids),
        "n_tokens": sum(lengths[i] for i in ids),
        "ids": ids,
    }, indent=2))
    save_cost(costs, out / f"{tag}_cost.json")
    print(f"{a.arm}: {len(ids)} examples, {sum(lengths[i] for i in ids)} tokens")
    print(json.dumps(costs, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
