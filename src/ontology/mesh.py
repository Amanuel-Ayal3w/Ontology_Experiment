"""
Parse the MeSH descriptor file into a concept table.

MeSH ships as an ASCII flat file (d2026.bin). Records are separated by
'*NEWRECORD'; each record is a block of 'KEY = VALUE' lines.

Fields used:
    UI     Unique identifier, e.g. D009203
    MH     Main Heading (preferred label)
    ENTRY  Synonyms, possibly '|'-delimited with lexical tags
    MN     Tree Number, e.g. C14.280.647.500

Depth comes free from the tree number: C14.280.647.500 has 4 segments,
so depth 4. This is why MeSH is preferable to SNOMED here -- no BFS
needed, and no polyhierarchy ambiguity to resolve.

Download (requires free NLM account):
    https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Concept:
    ui: str
    label: str
    tree_numbers: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """Shallowest hierarchy position. Top-level categories are depth 1."""
        if not self.tree_numbers:
            return 0
        return min(tn.count(".") + 1 for tn in self.tree_numbers)

    @property
    def categories(self) -> set[str]:
        """Top-level MeSH categories, e.g. {'C'} for Diseases."""
        return {tn[0] for tn in self.tree_numbers if tn}

    def ancestor_tree_numbers(self) -> set[str]:
        """All ancestor tree numbers, for concept propagation."""
        out: set[str] = set()
        for tn in self.tree_numbers:
            parts = tn.split(".")
            for i in range(1, len(parts)):
                out.add(".".join(parts[:i]))
        return out

    def all_terms(self) -> list[str]:
        return [self.label] + self.synonyms


def _clean_entry(raw: str) -> str:
    """ENTRY values look like 'Heart Attack|T047|...' -- keep the term."""
    return raw.split("|", 1)[0].strip()


def parse_mesh_ascii(path: str | Path) -> dict[str, Concept]:
    """Parse d20XX.bin into {ui: Concept}."""
    concepts: dict[str, Concept] = {}
    cur: dict[str, list[str]] = defaultdict(list)

    def flush() -> None:
        if not cur.get("UI") or not cur.get("MH"):
            return
        ui = cur["UI"][0]
        concepts[ui] = Concept(
            ui=ui,
            label=cur["MH"][0],
            tree_numbers=list(cur.get("MN", [])),
            synonyms=[
                _clean_entry(e)
                for e in cur.get("ENTRY", []) + cur.get("PRINT ENTRY", [])
            ],
        )

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("*NEWRECORD"):
                flush()
                cur = defaultdict(list)
                continue
            if " = " not in line:
                continue
            key, _, val = line.partition(" = ")
            cur[key.strip()].append(val.strip())
    flush()
    return concepts


def build_tree_index(concepts: dict[str, Concept]) -> dict[str, str]:
    """Map tree number -> UI, so ancestors can be resolved to concepts."""
    return {tn: c.ui for c in concepts.values() for tn in c.tree_numbers}


def restrict_to_categories(
    concepts: dict[str, Concept], categories: set[str]
) -> dict[str, Concept]:
    """
    Keep only concepts under given top-level categories.

    'C' = Diseases, 'D' = Chemicals and Drugs, 'E' = Techniques.
    Restricting cuts noise and shrinks |C_eff|, which raises k*.
    """
    return {ui: c for ui, c in concepts.items() if c.categories & categories}


def save(concepts: dict[str, Concept], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {ui: asdict(c) for ui, c in concepts.items()}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load(path: str | Path) -> dict[str, Concept]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {ui: Concept(**d) for ui, d in payload.items()}


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/mesh/d2026.bin"
    concepts = parse_mesh_ascii(src)
    print(f"parsed {len(concepts)} descriptors")
    by_depth: dict[int, int] = defaultdict(int)
    for c in concepts.values():
        by_depth[c.depth] += 1
    for d in sorted(by_depth):
        print(f"  depth {d}: {by_depth[d]}")


def parse_mesh_xml(path: str | Path) -> dict[str, Concept]:
    """Parse desc20XX.xml (or .gz) into {ui: Concept}.

    NLM archived the ASCII .bin format; current-year XML is the freely
    downloadable form. Same fields: DescriptorUI, DescriptorName,
    TreeNumberList, and all TermList strings as synonyms.
    """
    import gzip
    import xml.etree.ElementTree as ET

    opener = gzip.open if str(path).endswith(".gz") else open
    concepts: dict[str, Concept] = {}
    with opener(path, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "DescriptorRecord":
                continue
            ui = elem.findtext("DescriptorUI")
            label = elem.findtext("DescriptorName/String")
            if ui and label:
                terms = {
                    t.text.strip()
                    for t in elem.iterfind(".//TermList/Term/String")
                    if t.text and t.text.strip()
                }
                terms.discard(label)
                concepts[ui] = Concept(
                    ui=ui,
                    label=label,
                    tree_numbers=[
                        tn.text for tn in elem.iterfind("TreeNumberList/TreeNumber")
                        if tn.text
                    ],
                    synonyms=sorted(terms),
                )
            elem.clear()
    return concepts


def parse_mesh(path: str | Path) -> dict[str, Concept]:
    """Dispatch on format: .bin -> ASCII, .xml/.gz -> XML."""
    if str(path).endswith(".bin"):
        return parse_mesh_ascii(path)
    return parse_mesh_xml(path)
