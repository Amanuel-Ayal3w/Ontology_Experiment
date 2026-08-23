#!/usr/bin/env python3
"""GATE 2: measure saturation. This is Proof 1 and needs no GPU."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selection.coverage import measure_k_star

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="results/tags/tags.jsonl")
    ap.add_argument("--out", default="results/tags/kstar.json")
    ap.add_argument("--max-steps", type=int, default=8000)
    a = ap.parse_args()

    records = [json.loads(l) for l in open(a.tags)]
    ks = measure_k_star(records, max_steps=a.max_steps)

    print(f"C_eff                 = {ks['C_eff']}")
    print(f"mean tags/doc (m)     = {ks['mean_tags_per_doc']}")
    print(f"k* predicted (C/m)    = {ks['k_star_predicted']}")
    print(f"k* observed (elbow)   = {ks['k_star_observed']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(ks, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ks["gain_curve"]); ax[0].set_xlabel("step")
        ax[0].set_ylabel("marginal concepts gained"); ax[0].set_title("Marginal gain")
        if ks["k_star_observed"]:
            ax[0].axvline(ks["k_star_observed"], ls="--", c="r", label="observed k*")
            ax[0].legend()
        ax[1].plot(ks["coverage_curve"]); ax[1].set_xlabel("step")
        ax[1].set_ylabel("concepts covered"); ax[1].set_title("Cumulative coverage")
        plt.tight_layout(); plt.savefig(Path(a.out).parent / "kstar_curve.png", dpi=150)
        print("saved kstar_curve.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
