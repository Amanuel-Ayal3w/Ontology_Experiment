"""
Baseline selectors.

All selectors share the signature:
    select(records, lengths, budget_tokens, seed, **kw) -> list[int]

so train.py never knows which arm it is running. This is what guarantees
the arms differ ONLY in set membership.
"""

from __future__ import annotations

import random

from .coverage import coverage_select


def uniform_select(
    records: list[dict], lengths: list[int], budget_tokens: int, seed: int = 0, **_
) -> list[int]:
    """Random sample to token budget. The floor every method must beat."""
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    out, used = [], 0
    for i in order:
        if used + lengths[i] > budget_tokens and out:
            continue
        out.append(i)
        used += lengths[i]
        if used >= budget_tokens:
            break
    return out


def length_select(
    records: list[dict], lengths: list[int], budget_tokens: int, seed: int = 0, **_
) -> list[int]:
    """
    Longest examples first. A free heuristic.

    Included because coverage selection correlates with length, and this
    arm demonstrates the gain is not merely 'pick long documents'.
    """
    order = sorted(range(len(records)), key=lambda i: -lengths[i])
    out, used = [], 0
    for i in order:
        if used + lengths[i] > budget_tokens and out:
            continue
        out.append(i)
        used += lengths[i]
        if used >= budget_tokens:
            break
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def placebo_select(
    records: list[dict],
    lengths: list[int],
    budget_tokens: int,
    seed: int = 0,
    depths: dict[str, int] | None = None,
    **_,
) -> list[int]:
    """
    THE MECHANISM CONTROL. Run this in week 2, not week 4.

    Permutes which tag-set belongs to which document, preserving the tag
    distribution but destroying the correspondence. Then runs the ontology
    selector unchanged.

    If this matches the ontology arm, the observed gain came from the SHAPE
    of coverage-based selection (favouring long or unusual documents) rather
    than from actual concept coverage -- and the mechanism claim collapses
    regardless of whether ontology beat uniform.
    """
    rng = random.Random(seed + 9973)
    perm = list(range(len(records)))
    rng.shuffle(perm)
    shuffled = [
        {"tags": records[perm[i]]["tags"],
         "direct": records[perm[i]]["direct"],
         "n_chars": records[i]["n_chars"]}
        for i in range(len(records))
    ]
    return coverage_select(
        shuffled, lengths, budget_tokens, depths=depths, seed=seed
    )


def d4_select(
    records: list[dict],
    lengths: list[int],
    budget_tokens: int,
    seed: int = 0,
    texts: list[str] | None = None,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_clusters: int = 100,
    dedup_threshold: float = 0.95,
    device: str = "cuda",
    **_,
) -> list[int]:
    """
    D4-style embedding selection: encode -> cluster -> dedup -> sample.

    THE COMPETITOR. Requires a GPU forward pass over the whole corpus --
    that pass is precisely the cost this project asks whether we can avoid.
    Time it (see src/cost.py) on the SAME corpus and machine as the
    symbolic tagger; citing D4's published GPU-hours is not a comparison.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    if texts is None:
        raise ValueError("d4_select requires raw texts to encode")

    model = SentenceTransformer(encoder_name, device=device)
    emb = model.encode(
        texts, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    km = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=seed, n_init=4)
    labels = km.fit_predict(emb)

    # SemDeDup: within each cluster drop near-duplicates
    keep: list[int] = []
    for c in range(labels.max() + 1):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        # distance to centroid; keep the furthest (most prototypical dropped)
        sims = emb[idx] @ km.cluster_centers_[c]
        order = idx[np.argsort(sims)]
        kept_c: list[int] = []
        for i in order:
            if all(float(emb[i] @ emb[j]) < dedup_threshold for j in kept_c):
                kept_c.append(int(i))
        keep.extend(kept_c)

    # Round-robin across clusters for diversity
    by_cluster: dict[int, list[int]] = {}
    for i in keep:
        by_cluster.setdefault(int(labels[i]), []).append(i)
    rng = random.Random(seed)
    for v in by_cluster.values():
        rng.shuffle(v)

    out, used = [], 0
    cluster_ids = sorted(by_cluster)
    pos = {c: 0 for c in cluster_ids}
    while used < budget_tokens:
        progressed = False
        for c in cluster_ids:
            if pos[c] >= len(by_cluster[c]):
                continue
            i = by_cluster[c][pos[c]]
            pos[c] += 1
            progressed = True
            if used + lengths[i] > budget_tokens and out:
                continue
            out.append(i)
            used += lengths[i]
            if used >= budget_tokens:
                break
        if not progressed:
            break

    rng.shuffle(out)
    return out
