"""
Evaluation, stratified by concept frequency.

The head/tail split is the MECHANISM EVIDENCE, not a nice-to-have. If the
method works, gains concentrate on tail concepts -- rare content that
uniform sampling misses. A flat aggregate gain with no head/tail structure
suggests something other than concept coverage is driving the result.

Stratification uses the tagger on the EVAL set. That is analysis, not
selection -- keep the code path separate so no selection signal can leak
into evaluation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch


def concept_frequencies(pool_records: list[dict]):
    """Concept -> count over the candidate pool."""
    from collections import Counter
    freq = Counter()
    for r in pool_records:
        freq.update(r["tags"])
    return freq


def build_frequency_strata(
    pool_records: list[dict], head_quantile: float = 0.90
) -> tuple[set[str], set[str]]:
    """
    Partition concepts into head/tail by frequency in the CANDIDATE POOL.

    Head = top decile by frequency. Tail = bottom half. Concepts in between
    are excluded from both to sharpen the contrast.
    """
    freq: Counter = Counter()
    for r in pool_records:
        freq.update(r["tags"])
    if not freq:
        return set(), set()
    counts = sorted(freq.values())
    hi = counts[int(len(counts) * head_quantile)]
    lo = counts[int(len(counts) * 0.50)]
    head = {c for c, n in freq.items() if n >= hi}
    tail = {c for c, n in freq.items() if n <= lo}
    return head, tail


def stratify_eval(
    eval_items: list[dict], tagger, freq
) -> dict[str, list[int]]:
    """Median split on the pool frequency of each item's rarest matched concept.

    The quantile-set rule (tail = concepts with pool freq <= p10) yields an
    empty tail stratum on MedExQA: freq-1 concepts never appear in exam
    questions. Rarest-concept median split gives balanced, usable strata and
    is still "stratification by concept frequency in the candidate pool".
    Decided before any training results existed.
    """
    import statistics

    rarest: list[int | None] = []
    for item in eval_items:
        text = item["question"] + " " + " ".join(item.get("options", []))
        tags = tagger.tag(text)
        rarest.append(min((freq[t] for t in tags if t in freq), default=None))

    tagged = [r for r in rarest if r is not None]
    med = statistics.median(tagged) if tagged else 0
    strata: dict[str, list[int]] = {"head": [], "tail": [], "other": []}
    for i, r in enumerate(rarest):
        if r is None:
            strata["other"].append(i)
        elif r <= med:
            strata["tail"].append(i)
        else:
            strata["head"].append(i)
    return strata


@torch.no_grad()
def score_multiple_choice(model, tok, question: str, options: list[str]) -> int:
    """
    Score each option by mean log-likelihood of its continuation.

    Length-normalised, otherwise short options win systematically.
    """
    device = next(model.parameters()).device
    scores = []
    for opt in options:
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        full = prompt + opt
        p_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        f_ids = tok(full, return_tensors="pt").input_ids.to(device)
        logits = model(f_ids).logits[:, :-1]
        targets = f_ids[:, 1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        picked = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        cont = picked[:, p_ids.shape[1] - 1:]
        scores.append(cont.mean().item())
    return int(max(range(len(options)), key=lambda i: scores[i]))


def evaluate(
    model, tok, eval_items: list[dict], strata: dict[str, list[int]]
) -> dict:
    """Returns overall and per-stratum accuracy."""
    correct = [0] * len(eval_items)
    for i, item in enumerate(eval_items):
        pred = score_multiple_choice(model, tok, item["question"], item["options"])
        correct[i] = int(pred == item["answer_idx"])

    def acc(idxs: list[int]) -> float | None:
        if not idxs:
            return None
        return round(sum(correct[i] for i in idxs) / len(idxs), 4)

    return {
        "accuracy_overall": round(sum(correct) / max(len(correct), 1), 4),
        "accuracy_head": acc(strata["head"]),
        "accuracy_tail": acc(strata["tail"]),
        "accuracy_other": acc(strata["other"]),
        "n_head": len(strata["head"]),
        "n_tail": len(strata["tail"]),
        "n_total": len(eval_items),
    }


def aggregate_runs(runs_dir: str | Path) -> dict:
    """Collect metrics across arms and seeds; report mean +/- std."""
    import numpy as np

    runs_dir = Path(runs_dir)
    by_arm: dict[str, list[dict]] = {}
    for mf in runs_dir.rglob("metrics.json"):
        m = json.loads(mf.read_text())
        by_arm.setdefault(m["arm"], []).append(m)

    summary = {}
    for arm, ms in by_arm.items():
        row = {"n_seeds": len(ms)}
        for key in ("accuracy_overall", "accuracy_head", "accuracy_tail"):
            vals = [m[key] for m in ms if m.get(key) is not None]
            if vals:
                row[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }
        summary[arm] = row
    return summary


def paired_test(runs_dir: str | Path, arm_a: str, arm_b: str, key: str = "accuracy_tail"):
    """
    Paired t-test across seeds. Differences within seed variance are NULL.

    A 1.5-point gain with 2-point variance is not a result, and reporting
    it as one is the most common way these studies fail.
    """
    import numpy as np
    from scipy.stats import ttest_rel

    runs_dir = Path(runs_dir)
    a, b = {}, {}
    for mf in runs_dir.rglob("metrics.json"):
        m = json.loads(mf.read_text())
        if m.get(key) is None:
            continue
        if m["arm"] == arm_a:
            a[m["seed"]] = m[key]
        elif m["arm"] == arm_b:
            b[m["seed"]] = m[key]

    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return {"error": f"need >=2 shared seeds, got {len(seeds)}"}

    va = [a[s] for s in seeds]
    vb = [b[s] for s in seeds]
    t, p = ttest_rel(va, vb)
    return {
        "arm_a": arm_a, "arm_b": arm_b, "metric": key, "seeds": seeds,
        "mean_a": round(float(np.mean(va)), 4),
        "mean_b": round(float(np.mean(vb)), 4),
        "diff": round(float(np.mean(va) - np.mean(vb)), 4),
        "t": round(float(t), 3), "p": round(float(p), 4),
        "significant_at_05": bool(p < 0.05),
    }
