#!/usr/bin/env python3
"""
GATE 3: the mechanism check. Run in WEEK 2, not week 4.

If placebo matches ontology, the gain came from the SHAPE of coverage
selection rather than actual concept coverage -- and everything downstream
is uninterpretable. Better to learn this with three weeks left.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import aggregate_runs, paired_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--metric", default="accuracy_tail")
    a = ap.parse_args()

    print(json.dumps(aggregate_runs(a.runs), indent=2))
    print("\n--- ontology vs uniform (does it beat the floor?) ---")
    r1 = paired_test(a.runs, "ontology", "uniform", a.metric)
    print(json.dumps(r1, indent=2))
    print("\n--- ontology vs placebo (is the mechanism real?) ---")
    r2 = paired_test(a.runs, "ontology", "placebo", a.metric)
    print(json.dumps(r2, indent=2))

    if "error" in r2:
        print("\ninsufficient runs for the gate")
        return 1
    if not r2.get("significant_at_05") or r2.get("diff", 0) <= 0:
        print("\n*** GATE FAILED: ontology ~= placebo.")
        print("*** The effect is not attributable to concept coverage.")
        print("*** Stop and rethink before spending further compute.")
        return 1
    print("\nGATE PASSED: effect is attributable to concept coverage")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
