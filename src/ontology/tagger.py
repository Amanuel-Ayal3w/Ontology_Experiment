"""
Symbolic concept tagger.

This is the component the whole project rests on: if tagging recall is low,
there is no signal for selection to exploit, and no amount of downstream
care will recover it. Run `scripts/01_tag.py` diagnostics before proceeding.

Design constraints:
  * No forward passes. Aho-Corasick string matching only. This is what makes
    the selection cost claim hold.
  * Ambiguity filters are crude by necessity -- a disambiguation model would
    reintroduce the GPU cost we are trying to avoid. We report the discard
    rate as an honest limitation rather than hiding it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import ahocorasick

from .mesh import Concept, build_tree_index

# Terms that match constantly but carry no discriminative signal. MeSH entry
# terms include many of these ('Cold', 'Range', 'Character'). Extend freely --
# every addition should be logged in the diagnostics output.
STOP_TERMS = {
    "cold", "range", "character", "control", "male", "female", "human",
    "animal", "adult", "child", "infant", "aged", "time", "history",
    "pressure", "rate", "gas", "ion", "acid", "base", "salt", "water",
    "light", "sound", "color", "state", "form", "growth", "aging",
    "attention", "memory", "mood", "affect", "sensation", "movement",
    "reading", "speech", "learning", "judgment", "will", "self", "role",
    "risk", "safety", "quality", "value", "weight", "length", "volume",
    "mass", "area", "surface", "type", "class", "group", "family",
    "order", "life", "death", "birth", "food", "drink", "sleep", "rest",
    "work", "play", "art", "music", "dance", "religion", "law", "ethics",
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s-]")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


@dataclass
class TaggerStats:
    n_terms_total: int
    n_terms_kept: int
    n_terms_dropped_short: int
    n_terms_dropped_stop: int

    @property
    def discard_rate(self) -> float:
        dropped = self.n_terms_dropped_short + self.n_terms_dropped_stop
        return dropped / max(self.n_terms_total, 1)


class ConceptTagger:
    """Dictionary-based concept recognition over a MeSH concept table."""

    def __init__(
        self,
        concepts: dict[str, Concept],
        min_term_len: int = 4,
        stop_terms: set[str] | None = None,
    ):
        self.concepts = concepts
        self.tree_index = build_tree_index(concepts)
        self.min_term_len = min_term_len
        self.stop_terms = stop_terms if stop_terms is not None else STOP_TERMS

        self.automaton = ahocorasick.Automaton()
        self.stats = self._build()

    def _build(self) -> TaggerStats:
        total = kept = short = stop = 0
        # term -> set of UIs (a surface form can map to several concepts)
        term_map: dict[str, set[str]] = {}

        for ui, c in self.concepts.items():
            for term in c.all_terms():
                total += 1
                norm = normalise(term)
                if len(norm) < self.min_term_len:
                    short += 1
                    continue
                if norm in self.stop_terms:
                    stop += 1
                    continue
                term_map.setdefault(norm, set()).add(ui)
                kept += 1

        for term, uis in term_map.items():
            self.automaton.add_word(term, (term, tuple(uis)))
        self.automaton.make_automaton()

        return TaggerStats(total, kept, short, stop)

    @staticmethod
    def _is_word_boundary(text: str, start: int, end: int) -> bool:
        """Reject matches inside larger words ('mi' in 'mitral')."""
        before_ok = start == 0 or not text[start - 1].isalnum()
        after_ok = end >= len(text) - 1 or not text[end + 1].isalnum()
        return before_ok and after_ok

    def tag(self, text: str) -> set[str]:
        """Return direct concept UIs mentioned in text."""
        norm = normalise(text)
        found: set[str] = set()
        for end, (term, uis) in self.automaton.iter(norm):
            start = end - len(term) + 1
            if self._is_word_boundary(norm, start, end):
                found.update(uis)
        return found

    def propagate(self, uis: set[str]) -> set[str]:
        """Expand a concept set with all taxonomic ancestors."""
        out = set(uis)
        for ui in uis:
            c = self.concepts.get(ui)
            if not c:
                continue
            for atn in c.ancestor_tree_numbers():
                anc = self.tree_index.get(atn)
                if anc:
                    out.add(anc)
        return out

    def tag_corpus(
        self, texts: list[str], propagate: bool = True
    ) -> list[dict]:
        """
        Tag a corpus. Returns one record per text:
            {"direct": [...], "tags": [...], "n_chars": int}

        'tags' is the set used by selection -- propagated if requested,
        otherwise identical to 'direct'. The flat/propagated distinction
        is an ablation (does hierarchy help beyond a flat vocabulary?).
        """
        records = []
        for t in texts:
            direct = self.tag(t)
            tags = self.propagate(direct) if propagate else direct
            records.append(
                {
                    "direct": sorted(direct),
                    "tags": sorted(tags),
                    "n_chars": len(t),
                }
            )
        return records


def corpus_diagnostics(records: list[dict]) -> dict:
    """
    Gate metrics. Proceed only if:
        mean_tags_per_doc >= 3   (else too little signal)
        zero_tag_rate     < 0.20 (else tagger too strict)
    """
    n = len(records)
    counts = [len(r["tags"]) for r in records]
    direct_counts = [len(r["direct"]) for r in records]
    all_tags: Counter = Counter()
    for r in records:
        all_tags.update(r["tags"])

    c_eff = len(all_tags)
    m = sum(counts) / max(n, 1)

    return {
        "n_docs": n,
        "mean_tags_per_doc": round(m, 2),
        "mean_direct_per_doc": round(sum(direct_counts) / max(n, 1), 2),
        "zero_tag_rate": round(sum(1 for c in counts if c == 0) / max(n, 1), 3),
        "C_eff": c_eff,
        "k_star_predicted": round(c_eff / m) if m > 0 else None,
        "gate_density_ok": m >= 3.0,
        "gate_coverage_ok": (sum(1 for c in counts if c == 0) / max(n, 1)) < 0.20,
    }
