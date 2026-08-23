#!/usr/bin/env python3
"""Regenerate the paper's 'Training comparisons' block from current run data.

Idempotent: replaces everything between the BEGIN/END markers in paper.latex.
"""
import glob, json, statistics, subprocess, sys
from pathlib import Path

VALID_HASH = "dcb6259f87b2"
BEGIN = "% BEGIN AUTOGEN training-comparisons"
END = "% END AUTOGEN training-comparisons"

def main():
    by_arm = {}
    for f in glob.glob("results/runs/*/metrics.json"):
        d = json.load(open(f))
        if d.get("config_hash") == VALID_HASH:
            by_arm.setdefault(d["arm"], {})[d["seed"]] = d
    if not by_arm:
        print("no valid runs", file=sys.stderr); return 1

    def cell(arm, key):
        vals = [by_arm[arm][s][key] for s in sorted(by_arm[arm])
                if by_arm[arm][s].get(key) is not None]
        if not vals: return "--"
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"${m:.3f} \\pm {s:.3f}$" if len(vals) > 1 else f"${m:.3f}$"

    order = [a for a in ("uniform","length","ontology","placebo","d4") if a in by_arm]
    rows = []
    for arm in order:
        n = len(by_arm[arm])
        name = r"\textbf{Ontology}" if arm == "ontology" else arm.capitalize()
        rows.append(f"{name} & {n} & {cell(arm,'accuracy_overall')} & "
                    f"{cell(arm,'accuracy_head')} & {cell(arm,'accuracy_tail')} \\\\")

    # paired ontology vs placebo
    o, p = by_arm.get("ontology", {}), by_arm.get("placebo", {})
    shared = sorted(set(o) & set(p))
    diffs = [o[s]["accuracy_tail"] - p[s]["accuracy_tail"] for s in shared]
    if len(shared) >= 2:
        try:
            from scipy.stats import ttest_rel
            _, pv = ttest_rel([o[s]["accuracy_tail"] for s in shared],
                              [p[s]["accuracy_tail"] for s in shared])
            pv_s = f"$p={pv:.3f}$"
        except Exception:
            pv_s = "test unavailable"
        per = ", ".join(f"${d:+.4f}$" for d in diffs)
        consistent = all(d < 0 for d in diffs) or all(d > 0 for d in diffs)
        signnote = ("The sign is consistent" if consistent
                    else "The sign is not consistent across seeds")
        gate = (f"Across the {len(shared)} paired seeds the ontology--placebo "
                f"difference in tail accuracy was {per} "
                f"(mean ${statistics.mean(diffs):+.4f}$, {pv_s}). {signnote}, and "
                f"the difference is far from significance: the ontology arm was "
                f"not distinguishable from a selector using deliberately "
                f"scrambled tags, and the pre-registered mechanism gate "
                f"therefore fails---not because the placebo won, but because "
                f"nothing separates them.")
    elif len(shared) == 1:
        gate = (f"Only one paired seed is available; the observed "
                f"ontology--placebo tail difference is ${diffs[0]:+.4f}$, "
                f"which no test can distinguish from noise.")
    else:
        gate = "No paired ontology/placebo seeds are available."

    # ontology vs uniform -- H1's own test
    u = by_arm.get("uniform", {})
    sh_ou = sorted(set(o) & set(u))
    d_ou = [o[s]["accuracy_tail"] - u[s]["accuracy_tail"] for s in sh_ou]
    if len(sh_ou) >= 2:
        try:
            from scipy.stats import ttest_rel
            _, pv2 = ttest_rel([o[s]["accuracy_tail"] for s in sh_ou],
                               [u[s]["accuracy_tail"] for s in sh_ou])
            pv2_s = f"$p={pv2:.2f}$"
        except Exception:
            pv2_s = "test unavailable"
        h1 = (f" On the H1 comparison itself---coverage selection against uniform "
              f"sampling on tail accuracy---the paired difference is "
              f"${statistics.mean(d_ou):+.4f}$ ({pv2_s}), with the per-seed values "
              f"{', '.join(f'${d:+.4f}$' for d in d_ou)} differing in sign. H1 is "
              f"not supported: at this budget the ontology arm is indistinguishable "
              f"from random selection on the stratum it was predicted to improve.")
    else:
        h1 = ""

    u_mean = (statistics.mean([u[s]["accuracy_overall"] for s in sorted(u)])
              if u else float("nan"))
    seedline = ", ".join(f"{a}: $n={len(by_arm[a])}$" for a in order)

    body = f"""{BEGIN}
\\subsection{{Training comparisons}}

Table~\\ref{{tab:arms}} and Fig.~\\ref{{fig:arms}} report accuracy by arm at\nthe 1M-token budget, mean
$\\pm$ standard deviation across seeds ({seedline}). All arms share one base
checkpoint, one LoRA configuration, 500 optimiser steps, and the configuration
hash recorded in every run record.

\\begin{{table}}[h]\\centering\\small
\\begin{{tabular}}{{@{{}}lrccc@{{}}}}
\\toprule
\\textbf{{Arm}} & \\textbf{{n}} & \\textbf{{Overall}} & \\textbf{{Head}} & \\textbf{{Tail}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Accuracy by arm, stratified by the pool frequency of each item's
rarest matched concept. Zero-shot base accuracy is $0.423$.}}
\\label{{tab:arms}}
\\end{{table}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{results/figures/arms.png}}
\\caption{{Accuracy by arm and stratum. Bars give the mean over seeds with
standard deviation; overlaid dots are individual seeds. The seed spread
exceeds the between-arm differences in every stratum. In the tail
stratum---the one the method targets---the ontology and placebo arms are
indistinguishable, and both sit within roughly one standard deviation of
uniform sampling.}}
\\label{{fig:arms}}
\\end{{figure}}

Both selection arms improve slightly on uniform sampling in aggregate
accuracy. {gate}{h1}

We note also that uniform sampling at this budget does not improve on the
zero-shot base model (${u_mean:.3f}$ against $0.423$): at a $1.4\\%$ keep
ratio, randomly chosen instruction data contributes nothing measurable, which
is the regime in which selection was expected to matter most.
{END}"""

    src = Path("paper.latex").read_text()
    if BEGIN in src:
        pre = src.split(BEGIN)[0]
        post = src.split(END)[1]
        src = pre + body + post
    else:
        anchor = "\\subsection{Training comparisons}\n\\emph{To be completed.}"
        if anchor not in src:
            print("anchor not found", file=sys.stderr); return 1
        src = src.replace(anchor, body)
    Path("paper.latex").write_text(src)
    print("paper.latex training-comparisons updated")
    print(f"arms: {seedline}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
