#!/usr/bin/env python3
"""Cost table + quality/selection-cost frontier from run and cost records."""
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default="results/selections")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--tags", default="results/tags/tagging_cost.json")
    ap.add_argument("--out", default="results/frontier")
    a = ap.parse_args()

    # selection cost per arm (mean over seeds), tagging amortized onto ontology/placebo
    tag_cost = 0.0
    if Path(a.tags).exists():
        tc = json.load(open(a.tags))
        tag_cost = sum(v["cpu_seconds"] for v in tc.values())
    sel = {}
    for f in glob.glob(f"{a.selections}/*_cost.json"):
        arm = Path(f).name.split("_")[0]
        c = list(json.load(open(f)).values())[0]
        sel.setdefault(arm, []).append(c)
    cost_rows = {}
    for arm, cs in sel.items():
        cpu = sum(c["cpu_seconds"] for c in cs) / len(cs)
        wall = sum(c["wall_seconds"] for c in cs) / len(cs)
        gpu = sum(c.get("gpu_seconds", 0) for c in cs) / len(cs)
        if arm in ("ontology", "placebo"):
            cpu += tag_cost; wall += tag_cost
        cost_rows[arm] = {"cpu_s": round(cpu, 1), "wall_s": round(wall, 1),
                          "gpu_s": round(gpu, 1)}

    # quality per arm (mean over seeds)
    qual = {}
    for f in glob.glob(f"{a.runs}/*/metrics.json"):
        m = json.load(open(f))
        if m.get("accuracy_overall") is not None:
            qual.setdefault(m["arm"], []).append(m)
    table = {}
    for arm in sorted(set(cost_rows) | set(qual)):
        row = dict(cost_rows.get(arm, {}))
        ms = qual.get(arm, [])
        if ms:
            for k in ("accuracy_overall", "accuracy_head", "accuracy_tail"):
                vals = [m[k] for m in ms if m.get(k) is not None]
                if vals:
                    row[k] = round(sum(vals) / len(vals), 4)
            row["n_seeds"] = len(ms)
        table[arm] = row

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "frontier.json").write_text(json.dumps(table, indent=2))
    hdr = f"{'arm':10} {'cpu_s':>8} {'gpu_s':>8} {'acc':>7} {'head':>7} {'tail':>7} {'seeds':>5}"
    print(hdr); print("-" * len(hdr))
    for arm, r in table.items():
        print(f"{arm:10} {r.get('cpu_s','-'):>8} {r.get('gpu_s','-'):>8} "
              f"{r.get('accuracy_overall','-'):>7} {r.get('accuracy_head','-'):>7} "
              f"{r.get('accuracy_tail','-'):>7} {r.get('n_seeds','-'):>5}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for arm, r in table.items():
            if "accuracy_overall" not in r: continue
            x = max(r.get("cpu_s", 0) + 100 * r.get("gpu_s", 0), 0.05)  # gpu ~100x cpu price
            ax.scatter(x, r["accuracy_overall"], s=60)
            ax.annotate(arm, (x, r["accuracy_overall"]), xytext=(5, 5),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("selection cost (CPU-s + 100x GPU-s, log)")
        ax.set_ylabel("accuracy")
        ax.set_title("Quality vs selection cost")
        plt.tight_layout(); plt.savefig(out / "frontier.png", dpi=150)
        print(f"\nsaved {out}/frontier.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
