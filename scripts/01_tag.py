#!/usr/bin/env python3
"""GATE 1: tag the pool and check density. Stop if gates fail."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ontology.mesh import parse_mesh, restrict_to_categories, save as save_mesh
from src.ontology.tagger import ConceptTagger, corpus_diagnostics
from src.cost import measure, save as save_cost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="data/mesh/desc2026.gz")
    ap.add_argument("--pool", default="data/pool/pool.jsonl")
    ap.add_argument("--out", default="results/tags")
    ap.add_argument("--categories", default="C,D")
    ap.add_argument("--no-propagate", action="store_true")
    a = ap.parse_args()

    costs = {}
    concepts = parse_mesh(a.mesh)
    concepts = restrict_to_categories(concepts, set(a.categories.split(",")))
    print(f"concepts after category restriction: {len(concepts)}")

    corpus = [json.loads(l) for l in open(a.pool, encoding="utf-8")]
    texts = [f"{r.get('instruction','')} {r.get('input','')} {r.get('output','')}"
             for r in corpus]
    print(f"pool size: {len(texts)}")

    with measure("build_automaton", costs):
        tagger = ConceptTagger(concepts)
    print(f"terms kept={tagger.stats.n_terms_kept} "
          f"discard_rate={tagger.stats.discard_rate:.3f}")

    with measure("tag_corpus", costs):
        records = tagger.tag_corpus(texts, propagate=not a.no_propagate)

    diag = corpus_diagnostics(records)
    diag["tagger_discard_rate"] = round(tagger.stats.discard_rate, 3)
    print(json.dumps(diag, indent=2))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "tags.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    save_mesh(concepts, out / "concepts.json")
    save_cost(costs, out / "tagging_cost.json")

    if not diag["gate_density_ok"]:
        print("\n*** GATE FAILED: mean tags/doc < 3. Loosen filters or change corpus. ***")
        return 1
    if not diag["gate_coverage_ok"]:
        print("\n*** GATE FAILED: zero-tag rate >= 20%. Tagger too strict. ***")
        return 1
    print("\nGATES PASSED -> proceed to 03_kstar.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
