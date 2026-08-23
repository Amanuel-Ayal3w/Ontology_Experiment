"""Smoke test on a synthetic MeSH fragment. Run: python tests/smoke_test.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ontology.mesh import Concept, parse_mesh_ascii
from src.ontology.tagger import ConceptTagger, corpus_diagnostics
from src.selection.coverage import coverage_select, measure_k_star
from src.selection.baselines import uniform_select, length_select, placebo_select

MESH_SAMPLE = """*NEWRECORD
RECTYPE = D
MH = Myocardial Infarction
ENTRY = Heart Attack|T047|NON|EQV
ENTRY = Myocardial Infarct
ENTRY = Cardiac Infarction
MN = C14.280.647.500
UI = D009203

*NEWRECORD
RECTYPE = D
MH = Hypertension
ENTRY = High Blood Pressure
MN = C14.907.489
UI = D006973

*NEWRECORD
RECTYPE = D
MH = Heart Failure
ENTRY = Cardiac Failure
MN = C14.280.238
UI = D006333

*NEWRECORD
RECTYPE = D
MH = Myocardial Ischemia
MN = C14.280.647
UI = D017202

*NEWRECORD
RECTYPE = D
MH = Heart Diseases
MN = C14.280
UI = D006331

*NEWRECORD
RECTYPE = D
MH = Cardiovascular Diseases
MN = C14
UI = D002318

*NEWRECORD
RECTYPE = D
MH = Takotsubo Cardiomyopathy
ENTRY = Broken Heart Syndrome
MN = C14.280.238.900
UI = D054549
"""

DOCS = [
    "A 54-year-old presented with chest pain. ECG confirmed myocardial infarction.",
    "Following a heart attack, secondary prevention with aspirin is recommended.",
    "Patient with acute myocardial infarct developed cardiogenic shock.",
    "Management of hypertension in patients with heart failure requires care.",
    "Takotsubo cardiomyopathy mimics acute coronary syndrome presentation.",
    "The weather today is pleasant and unrelated to any clinical topic.",
]


def main() -> int:
    tmp = Path("/tmp/mesh_sample.bin")
    tmp.write_text(MESH_SAMPLE, encoding="utf-8")

    concepts = parse_mesh_ascii(tmp)
    assert len(concepts) == 7, f"expected 7 concepts, got {len(concepts)}"
    mi = concepts["D009203"]
    assert mi.depth == 4, f"MI depth should be 4, got {mi.depth}"
    assert mi.categories == {"C"}
    print(f"[mesh]    parsed {len(concepts)} concepts; MI depth={mi.depth} OK")

    tagger = ConceptTagger(concepts)
    print(f"[tagger]  terms kept={tagger.stats.n_terms_kept} "
          f"discard_rate={tagger.stats.discard_rate:.2f}")

    # The key claim: three different surface forms -> same concept
    d0 = tagger.tag(DOCS[0])
    d1 = tagger.tag(DOCS[1])
    d2 = tagger.tag(DOCS[2])
    assert "D009203" in d0 and "D009203" in d1 and "D009203" in d2, \
        "synonym matching failed"
    print("[tagger]  'myocardial infarction' / 'heart attack' / "
          "'myocardial infarct' -> same concept OK")

    # Propagation
    prop = tagger.propagate({"D009203"})
    assert "D017202" in prop and "D006331" in prop and "D002318" in prop, \
        f"ancestor propagation failed: {prop}"
    print(f"[tagger]  propagation D009203 -> {len(prop)} concepts OK")

    records = tagger.tag_corpus(DOCS, propagate=True)
    diag = corpus_diagnostics(records)
    print(f"[diag]    {diag}")

    lengths = [max(len(d.split()), 1) for d in DOCS]

    ks = measure_k_star(records)
    print(f"[k*]      predicted={ks['k_star_predicted']} "
          f"observed={ks['k_star_observed']} C_eff={ks['C_eff']}")
    print(f"[k*]      gain curve: {ks['gain_curve']}")

    budget = sum(lengths) // 2
    sel_cov = coverage_select(records, lengths, budget, seed=0)
    sel_uni = uniform_select(records, lengths, budget, seed=0)
    sel_len = length_select(records, lengths, budget, seed=0)
    sel_pla = placebo_select(records, lengths, budget, seed=0)

    for name, sel in [("coverage", sel_cov), ("uniform", sel_uni),
                      ("length", sel_len), ("placebo", sel_pla)]:
        toks = sum(lengths[i] for i in sel)
        assert toks <= budget, f"{name} exceeded budget"
        print(f"[select]  {name:9s} n={len(sel)} tokens={toks}/{budget} ids={sorted(sel)}")

    # Coverage arm should reach more distinct concepts than uniform, on average
    cov_concepts = set()
    for i in sel_cov:
        cov_concepts.update(records[i]["tags"])
    uni_concepts = set()
    for i in sel_uni:
        uni_concepts.update(records[i]["tags"])
    print(f"[select]  concepts covered: coverage={len(cov_concepts)} "
          f"uniform={len(uni_concepts)}")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
