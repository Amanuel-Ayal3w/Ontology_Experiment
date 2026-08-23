"""
Ontological coverage selection.

The objective is monotone submodular, so greedy attains the standard
(1 - 1/e) approximation.

Two design decisions matter and are easy to get wrong:

1. DIMINISHING RETURNS, not binary coverage.
   Binary coverage saturates: once every concept is covered, marginal gain
   is zero for all remaining candidates and selection degenerates to random.
   Saturation occurs near k* = |C_eff| / m. Using w/(1+n_c) keeps the
   objective informative past k*.

2. PER-TOKEN NORMALISATION.
   Without dividing by length, the objective just picks the longest
   documents (more concepts per document = higher raw gain). Since the
   budget is denominated in tokens, gain-per-token is the correct
   quantity.

The returned set is ALWAYS shuffled. Greedy emits documents ordered by
marginal gain; preserving that order would introduce an implicit curriculum
and confound selection with ordering.
"""

from __future__ import annotations

import heapq
import random
from collections import defaultdict


def _weights(
    records: list[dict], depths: dict[str, int] | None, scheme: str
) -> dict[str, float]:
    """Concept weights. 'uniform' or 'depth' (deeper = more specific)."""
    if scheme == "uniform" or depths is None:
        return defaultdict(lambda: 1.0)
    w: dict[str, float] = defaultdict(lambda: 1.0)
    for r in records:
        for c in r["tags"]:
            w[c] = float(depths.get(c, 1))
    return w


def coverage_select(
    records: list[dict],
    lengths: list[int],
    budget_tokens: int,
    depths: dict[str, int] | None = None,
    weight_scheme: str = "uniform",
    seed: int = 0,
    lazy: bool = True,
) -> list[int]:
    """
    Greedy coverage-per-token selection.

    records: output of ConceptTagger.tag_corpus
    lengths: token count per record (same order)
    Returns shuffled list of selected indices.
    """
    w = _weights(records, depths, weight_scheme)
    n_c: dict[str, int] = defaultdict(int)
    selected: list[int] = []
    used = 0
    chosen = set()

    def gain(i: int) -> float:
        if lengths[i] <= 0:
            return 0.0
        g = sum(w[c] / (1 + n_c[c]) for c in records[i]["tags"])
        return g / lengths[i]

    if lazy:
        # Lazy greedy: submodularity means gains only decrease, so a stale
        # top-of-heap value that still beats the runner-up after refresh
        # is genuinely optimal. Large speedup on big corpora.
        heap = [(-gain(i), i, 0) for i in range(len(records))]
        heapq.heapify(heap)
        it = 0
        while heap and used < budget_tokens:
            neg_g, i, stamp = heapq.heappop(heap)
            if i in chosen:
                continue
            if stamp != it:
                heapq.heappush(heap, (-gain(i), i, it))
                continue
            if used + lengths[i] > budget_tokens and selected:
                continue
            selected.append(i)
            chosen.add(i)
            used += lengths[i]
            for c in records[i]["tags"]:
                n_c[c] += 1
            it += 1
    else:
        remaining = set(range(len(records)))
        while remaining and used < budget_tokens:
            best = max(remaining, key=gain)
            if used + lengths[best] > budget_tokens and selected:
                remaining.discard(best)
                continue
            selected.append(best)
            remaining.discard(best)
            used += lengths[best]
            for c in records[best]["tags"]:
                n_c[c] += 1

    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def measure_k_star(
    records: list[dict], max_steps: int = 20000, seed: int = 0
) -> dict:
    """
    Measure the saturation point empirically (Proof 1).

    Runs BINARY coverage greedy, recording marginal gain at each step.
    The step at which gain collapses is the observed k*. Compare against
    the predicted |C_eff| / m.

    Returns a dict with the gain curve and both k* estimates -- this is a
    reportable result on its own and requires no GPU.
    """
    covered: set[str] = set()
    chosen: set[int] = set()
    gains: list[int] = []
    cumulative: list[int] = []

    all_tags: set[str] = set()
    for r in records:
        all_tags.update(r["tags"])
    c_eff = len(all_tags)
    m = sum(len(r["tags"]) for r in records) / max(len(records), 1)

    steps = min(max_steps, len(records))
    for _ in range(steps):
        best_i, best_g = -1, -1
        for i, r in enumerate(records):
            if i in chosen:
                continue
            g = len(set(r["tags"]) - covered)
            if g > best_g:
                best_i, best_g = i, g
        if best_i < 0:
            break
        chosen.add(best_i)
        covered.update(records[best_i]["tags"])
        gains.append(best_g)
        cumulative.append(len(covered))
        if best_g == 0:
            break

    # Observed k*: first step where marginal gain hits zero, else where
    # cumulative coverage reaches 95% of C_eff.
    observed = None
    for i, g in enumerate(gains):
        if g == 0:
            observed = i
            break
    if observed is None:
        for i, cum in enumerate(cumulative):
            if cum >= 0.95 * c_eff:
                observed = i
                break

    return {
        "C_eff": c_eff,
        "mean_tags_per_doc": round(m, 2),
        "k_star_predicted": round(c_eff / m) if m > 0 else None,
        "k_star_observed": observed,
        "gain_curve": gains,
        "coverage_curve": cumulative,
    }
