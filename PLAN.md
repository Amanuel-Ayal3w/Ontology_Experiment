# PLAN — Ontological Coverage Selection

**Status 2026-08-24: the four-week experiment is complete.** Gates 0–2 passed,
gate 3 failed, the paper is written with real numbers, and the Overleaf bundle
builds. Everything below is either a record of what was decided, or optional
extension work.

For the result itself see `README.md`; for why it came out that way and what
not to re-litigate, see `HANDOFF.md`.

---

## What ran

| Stage | Where | Outcome |
|---|---|---|
| `prepare_data.py` | GPU box w/ internet | 167,102-example pool, 940-item MedExQA test set, both sha256-pinned |
| Gate 0 — contamination | GPU | base accuracy 0.423 — headroom, MedExQA kept (no switch to MedXpertQA) |
| Gate 1 — tagging density | local CPU | 23.04 tags/doc, 11.8% zero-tag, 0.1% discard — passed comfortably |
| Gate 2 — k\* | local CPU | predicted 575, observed exhaustion 3,523; budget ladder brackets both |
| Selection | local CPU | 4 arms × 3 seeds at the 1M budget, 42 CPU-s total, 0 GPU-s |
| Training + eval | Beam | 9 runs (uniform/ontology/placebo × 3 seeds), ~9 min each, config hash `dcb6259f87b2` |
| Gate 3 — placebo | local | **FAILED**: ontology − placebo = −0.007 tail accuracy, p = 0.71 |
| Analysis + paper | local | frontier, figures, autogen results block, `overleaf_flat.zip` |

Nine runs, one budget. That is the whole evidential base, and the paper is
explicit about it.

---

## Decisions taken, with their triggers

| Trigger | What we did |
|---|---|
| Gate 0 ≤ 0.60 | Kept MedExQA; no switch to MedXpertQA needed |
| Gate 1 comfortably passed | Kept the filters as written; no corpus change |
| Gate 2 k\* inside ladder | Kept `budgets_tokens` at 250k/500k/1M/2.5M; reported at 1M |
| Kaggle session limits + 9-hr kills | Abandoned the Kaggle plan for Beam (`beam_app.py`); `KAGGLE_NOTEBOOK.md` kept only for history |
| D4 container SIGSEGVs | Reported encoder cost analytically at `2P'T`, refused to splice a timing from other hardware |
| **Gate 3 failed** | Stopped the run matrix. Wrote it up as a diagnosed null rather than buying more budgets in the hope of a different answer |

The last row is the one that matters. Pulling gate 3 into week 2 rather than
week 4 is what made a clean write-up possible instead of a scramble.

---

## Remaining work, in descending order of value

Nothing here is required for the report to stand.

### 1. Budget sweep — the largest gap

H4 (advantage diminishes above k\*) has no evidence either way; only the 1M
budget was run. The sweep is also the only thing that could show the ontology
mattering at *some* budget rather than none.

- [ ] uniform / ontology / placebo × 3 seeds at 250k, 500k, 2.5M — 27 runs
- [ ] Regenerate with `09_write_results.py`, `10_figures.py`, `11_bundle.py`
- Cost: ~4 GPU-hours. Selection is free and already CPU-only.

### 2. `scripts/02_validate_tagger.py` — cheap and closes a promise

The paper commits to tagger P/R against a resource with human concept
annotations, "rather than by manual inspection of a small sample." That
measurement does not exist. Right now the tagging-quality argument rests on
density and discard-rate diagnostics, which is weaker than what was promised.

- [ ] Register for BioASQ, resolve the actual concept-field schema
- [ ] Write the script; it is CPU-only and needs no training

### 3. A working `d4` arm

Converts H2 from a FLOP argument into a measured one. Needs a GPU tier where the
container does not crash. If it runs, note the constraint the paper imposes:
**all selectors must be timed on the same machine**, so the CPU selectors need a
timing pass on that box too — a few CPU-minutes inside the same session.

### 4. The ablations — but reconsider them first

Saturation (binary vs diminishing returns), propagation (flat vs ancestor), and
weighting (uniform vs depth) are all supported by existing knobs
(`coverage_select`, `tag_corpus(propagate=...)`, `weight_scheme=`). All three
assume the concept signal contributes something. The placebo says it does not.
Asking whether *propagated* concept coverage beats *flat* concept coverage is
the wrong question when neither beats permuted tags.

The exception:

- [ ] **Ablation 5 — cost scaling.** Selection cost at several pool sizes,
      showing the cost ratio grows with corpus size while training cost does
      not. CPU-only, no ontology premise, and it directly strengthens the one
      quantitative result that survived. Do this one.
- [ ] Ablation 4 (exact / synonym-expanded / neural tagger) has no code path and
      is only interesting if the mechanism is revisited at another budget.

### 5. Paper hygiene before any submission

- [ ] Verify every arXiv ID resolves. Several were drafted from search snippets:
      `ofa` (arXiv:2605.26761), `survey` (arXiv:2606.10706), `evontree`
      (arXiv:2510.26683). A fabricated citation in a registered report is fatal.
- [x] `\bibitem{ost}` placeholder title resolved (arXiv:2605.07488).
- [x] Author, affiliation, and contact filled in.
- [ ] Read D4 §4.3 and arXiv:2503.22006 directly rather than via summary.

---

## Invariants — do not break when editing

- One shared selector signature; `train.py` never knows which arm it runs
- `results/selections/` holds ID lists only, so runs reproduce without MeSH
- Those IDs are raw line indices into `pool.jsonl` — a different pool file gives
  silently garbage arms with no error. The pool sha256 is recorded in every
  selection output; check it before trusting a run
- `max_steps` fixed, **never epochs**
- Every selector shuffles before returning
- Config hash in every `metrics.json`
- Eval-set tagging runs through a code path **separate from selection**
- Do not hand-edit between the `% BEGIN/END AUTOGEN` markers in `paper.latex` —
  `09_write_results.py` overwrites that block

---

## If the project is restarted rather than extended

The fallback identified during scoping is still the stronger idea and is
untouched by this null: **ontology-guided synthetic generation** — walk the
ontology and generate examples for uncovered concepts instead of selecting them
from a corpus. See `HANDOFF.md` §9 for why the refutation here does not carry
over to it.
