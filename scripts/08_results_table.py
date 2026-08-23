#!/usr/bin/env python3
"""Emit the LaTeX arm table and paired tests from valid run records only."""
import argparse, glob, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VALID_HASH = "dcb6259f87b2"

def load(runs_dir, want_hash):
    by_arm = {}
    for f in glob.glob(f"{runs_dir}/*/metrics.json"):
        d = json.load(open(f))
        if want_hash and d.get("config_hash") != want_hash:
            print(f"  skipping stale {d['arm']}/{d['seed']} "
                  f"(hash {d.get('config_hash')})", file=sys.stderr)
            continue
        by_arm.setdefault(d["arm"], {})[d["seed"]] = d
    return by_arm

def ms(vals):
    if not vals: return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--hash", default=VALID_HASH)
    ap.add_argument("--out", default="results/arm_table.tex")
    a = ap.parse_args()

    by_arm = load(a.runs, a.hash)
    order = [x for x in ("uniform", "length", "ontology", "placebo", "d4") if x in by_arm]

    rows = []
    print(f"{'arm':10} {'n':>2} {'overall':>16} {'head':>16} {'tail':>16}")
    for arm in order:
        seeds = sorted(by_arm[arm])
        cells = {}
        for key in ("accuracy_overall", "accuracy_head", "accuracy_tail"):
            m, s = ms([by_arm[arm][x][key] for x in seeds
                       if by_arm[arm][x].get(key) is not None])
            cells[key] = (m, s)
        print(f"{arm:10} {len(seeds):>2} "
              + " ".join(f"{cells[k][0]:.4f}+-{cells[k][1]:.4f}"
                         for k in ("accuracy_overall","accuracy_head","accuracy_tail")))
        rows.append((arm, len(seeds), cells))

    # paired tests on shared seeds
    print("\npaired comparisons (tail):")
    tests = {}
    for a_arm, b_arm in (("ontology","uniform"), ("ontology","placebo"), ("placebo","uniform")):
        if a_arm not in by_arm or b_arm not in by_arm: continue
        shared = sorted(set(by_arm[a_arm]) & set(by_arm[b_arm]))
        if len(shared) < 2:
            print(f"  {a_arm} vs {b_arm}: only {len(shared)} shared seed(s) -- no test")
            tests[(a_arm,b_arm)] = {"n": len(shared), "p": None}
            continue
        va = [by_arm[a_arm][s]["accuracy_tail"] for s in shared]
        vb = [by_arm[b_arm][s]["accuracy_tail"] for s in shared]
        diffs = [x-y for x, y in zip(va, vb)]
        try:
            from scipy.stats import ttest_rel
            t, p = ttest_rel(va, vb)
        except Exception:
            t, p = float("nan"), None
        print(f"  {a_arm} vs {b_arm}: n={len(shared)} diff={statistics.mean(diffs):+.4f} "
              f"per-seed={[f'{d:+.4f}' for d in diffs]} p={p}")
        tests[(a_arm,b_arm)] = {"n": len(shared), "diff": statistics.mean(diffs), "p": p}

    # LaTeX
    lines = [r"\begin{tabular}{@{}lrccc@{}}", r"\toprule",
             r"\textbf{Arm} & \textbf{n} & \textbf{Overall} & \textbf{Head} & \textbf{Tail} \\",
             r"\midrule"]
    for arm, n, cells in rows:
        name = r"\textbf{Ontology}" if arm == "ontology" else arm.capitalize()
        f = lambda k: (f"${cells[k][0]:.3f} \\pm {cells[k][1]:.3f}$"
                       if cells[k][0] is not None else "--")
        lines.append(f"{name} & {n} & {f('accuracy_overall')} & "
                     f"{f('accuracy_head')} & {f('accuracy_tail')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
