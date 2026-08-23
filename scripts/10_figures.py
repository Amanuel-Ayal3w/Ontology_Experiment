#!/usr/bin/env python3
"""Paper figures: arm comparison with seed scatter, and selection profile."""
import glob, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID = "dcb6259f87b2"
OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
COL = {"uniform": "#7a7a7a", "ontology": "#1f6f8b", "placebo": "#c2571a"}

def load():
    by = {}
    for f in glob.glob("results/runs/*/metrics.json"):
        d = json.load(open(f))
        if d.get("config_hash") == VALID:
            by.setdefault(d["arm"], {})[d["seed"]] = d
    return by

def fig_arms(by):
    """Grouped bars with individual seeds overlaid -- shows the variance honestly."""
    arms = [a for a in ("uniform", "ontology", "placebo") if a in by]
    metrics = [("accuracy_overall", "Overall"), ("accuracy_head", "Head"),
               ("accuracy_tail", "Tail")]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = 0.25
    for i, arm in enumerate(arms):
        seeds = sorted(by[arm])
        xs = [j + (i - (len(arms)-1)/2) * w for j in range(len(metrics))]
        means, sds = [], []
        for key, _ in metrics:
            vals = [by[arm][s][key] for s in seeds if by[arm][s].get(key) is not None]
            means.append(statistics.mean(vals))
            sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ax.bar(xs, means, w, yerr=sds, capsize=3, label=f"{arm} (n={len(seeds)})",
               color=COL.get(arm, "#999"), alpha=.85, edgecolor="white", zorder=2)
        for j, (key, _) in enumerate(metrics):
            for s in seeds:
                v = by[arm][s].get(key)
                if v is not None:
                    ax.plot(xs[j], v, "o", ms=3.5, color="black", alpha=.65, zorder=3)
    ax.axhline(0.423, ls="--", lw=1, color="crimson", zorder=1,
               label="zero-shot base (0.423)")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.38, 0.52)
    ax.set_title("Accuracy by arm and concept-frequency stratum\n"
                 "(bars: mean $\\pm$ sd; dots: individual seeds; chance $=0.25$)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), frameon=False)
    ax.grid(axis="y", alpha=.25, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "arms.png", dpi=200); plt.close()
    print("wrote results/figures/arms.png")

def fig_profile():
    """Why the placebo works: both tag-driven arms select short examples."""
    rows = []
    for f in sorted(glob.glob("results/selections/*_1000000_0.json")):
        if "_cost" in f: continue
        d = json.load(open(f))
        if d["arm"] == "length": continue
        rows.append((d["arm"], d["n_selected"], d["n_tokens"] / d["n_selected"]))
    order = {"uniform": 0, "ontology": 1, "placebo": 2}
    rows.sort(key=lambda r: order.get(r[0], 9))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    names = [r[0] for r in rows]
    cols = [COL.get(n, "#999") for n in names]
    a1.bar(names, [r[1] for r in rows], color=cols, alpha=.85, edgecolor="white")
    a1.set_ylabel("examples selected"); a1.set_title("Subset size at 1M tokens", fontsize=10)
    a2.bar(names, [r[2] for r in rows], color=cols, alpha=.85, edgecolor="white")
    a2.set_ylabel("mean tokens / example"); a2.set_title("Mean example length", fontsize=10)
    for ax in (a1, a2):
        ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "selection_profile.png", dpi=200); plt.close()
    print("wrote results/figures/selection_profile.png")

def fig_cost():
    """Selection cost in FLOPs against the training run it enables."""
    cm = json.load(open("results/frontier/cost_model.json"))
    enc, trn = cm["d4_encoder_flops_2PT"], cm["finetune_flops_6PT"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    labels = ["Ontology selection\n(CPU only)", "Embedding selection\n($2P'T$)",
              "LoRA fine-tune\n($6PT$)"]
    ys = range(3)
    ax.barh([1, 2], [enc, trn], color=["#c2571a", "#7a7a7a"], alpha=.85,
            edgecolor="white")
    # ontology uses no GPU FLOPs at all: mark it, do not draw a fake bar
    ax.plot([1e8], [0], marker="|", ms=14, mew=2.5, color="#1f6f8b")
    ax.text(1.4e8, 0, "0 GPU FLOPs  (42 CPU-seconds)", va="center", fontsize=8.5,
            color="#1f6f8b")
    ax.text(enc * 1.15, 1, f"{enc:.2e}", va="center", fontsize=8.5)
    ax.text(trn * 1.15, 2, f"{trn:.2e}", va="center", fontsize=8.5)
    ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(1e8, 2e17)
    ax.set_xlabel("FLOPs (log scale)")
    ax.set_title("Selection cost against the training run it enables", fontsize=10)
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "cost.png", dpi=200); plt.close()
    print("wrote results/figures/cost.png")

if __name__ == "__main__":
    by = load()
    fig_arms(by); fig_profile(); fig_cost()
