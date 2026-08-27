# Zero-GPU Data Selection for Instruction Tuning

This repository contains the pipeline, results, and paper for a registered report
testing whether **MeSH concept-coverage selection** outperforms uniform sampling
when fine-tuning a medical instruction model at a fixed token budget. Selection
runs on CPU only. The premise is a cost asymmetry: embedding-based selectors (D4
and its descendants) must push the entire candidate pool through a GPU encoder
before a single training step; this method does not.

The pre-registered mechanism gate **failed**. That is the finding. The design was
constructed so that a failure would identify its own cause.

---

## Headline result

At a 1M-token budget (a 1.4% keep ratio) on MedExQA, 3 seeds per arm:

| Arm | n | Overall | Head | Tail |
|---|---|---|---|---|
| Uniform | 3 | 0.421 ± 0.005 | 0.462 ± 0.005 | 0.441 ± 0.011 |
| **Ontology** | 3 | 0.428 ± 0.003 | 0.474 ± 0.002 | 0.456 ± 0.016 |
| Placebo (permuted tags) | 3 | 0.429 ± 0.004 | 0.466 ± 0.006 | 0.463 ± 0.018 |

Zero-shot base accuracy is 0.423.

- **Ontology vs. placebo, tail accuracy:** −0.007 (p = 0.71), sign inconsistent
  across seeds. Scrambling the tag→example correspondence changes nothing, so
  what the selector retains cannot be concept coverage. **Mechanism refuted.**
- **Ontology vs. uniform, tail accuracy (H1):** +0.016 (p = 0.40), also
  sign-inconsistent. **H1 unsupported.**
- **Operative variable:** the per-token objective
  `g(x) = (1/|x|) · Σ_c w(c)/(1+n_c)` rewards short, tag-dense examples. At the
  1M budget the ontology arm keeps 7,899 examples (mean 127 tokens) and the
  placebo 10,688 (mean 94), against uniform's 2,383 (mean 420). More, shorter
  examples at a fixed 500-step budget yields more distinct instructions per
  optimiser step. No ontology is required to produce this effect.

**Two findings survive the null:**

1. **The saturation bound locates the elbow, not exhaustion.** `k* = |C_eff|/m`
   predicts 575 examples; the empirical gain curve elbows there, but complete
   coverage arrives only at 3,523 (~6×), because greedy selection increasingly
   re-covers concepts it already holds.
2. **The cost asymmetry is real and larger than expected.** Tagging and selection
   over the 167k pool cost **42 CPU-seconds and zero GPU-seconds**. Encoding the
   same 73.9M tokens with a 22.7M-parameter MiniLM is 3.35e15 FLOPs at 2P'T —
   **36% of the 9.24e15 FLOPs of the LoRA fine-tune it is meant to make cheaper**.

The full argument, figures, and limitations are in `paper.latex`.

---

## Method in three operations

1. **Lexical grounding** — Aho–Corasick over MeSH 2026 labels and entry terms
   (categories C: Diseases, D: Chemicals/Drugs) → concept IDs. Terms under 4
   characters and common English words are dropped (0.1% discard rate).
2. **Taxonomic positioning** — depth from tree-number segment count
   (`C14.280.647.500` → depth 4); ancestors by truncating segments.
3. **Coverage maximisation** — greedy submodular on `g(x)` above, then shuffle.

No reasoner, no OWL axioms, no triples, no inference. The corpus is never
transformed: concept recognition produces a **stand-off annotation layer used
only for subset selection**, and the training data remains unmodified natural
language.

### Arms

| Arm | Rule | Selection cost | Role |
|---|---|---|---|
| `uniform` | random | none | floor — must be beaten |
| `length` | longest first | none | controls for "merely picks long documents" |
| `ontology` | coverage per token | CPU only | the method |
| `placebo` | **permuted tags**, same algorithm | CPU only | mechanism control |
| `d4` | encode → kmeans → semdedup → sample | 1 GPU pass | the competing method |

The placebo is the load-bearing control: it preserves the tag *distribution*
exactly while destroying the example↔concept correspondence.

---

## Repository layout

```
src/
  ontology/mesh.py        MeSH ASCII (d2026.bin) parser, tree-number depth, ancestors
  ontology/tagger.py      Aho-Corasick ConceptTagger, corpus tagging, propagation
  selection/coverage.py   greedy submodular coverage selector, k* measurement
  selection/baselines.py  uniform / length / placebo / d4 selectors
  selection/registry.py   one shared selector signature — train.py never knows the arm
  train.py                LoRA fine-tune (fixed max_steps, never epochs)
  evaluate.py             multiple-choice scoring, head/tail stratification, paired t-test
  cost.py                 wall-clock / CPU-second / FLOP accounting

scripts/                  the pipeline, in run order (00 … 11) — see below
configs/base.yaml         shared by ALL arms; only `arm` and `seed` may differ
tests/smoke_test.py       synthetic-MeSH end-to-end check (a script, not pytest)

beam_app.py               GPU stages on Beam (H100/A10G) — the runner used for all runs
modal_app.py              earlier Modal runner (superseded)
status.sh                 live dashboard: gates, Beam tasks, downloaded metrics

results/                  committed results (large intermediates are gitignored)
paper.latex               the registered report, results filled in
```

### What is and is not committed

Tracked: all code, configs, the paper, and the small result artifacts —
`results/runs/*/metrics.json`, `results/selections/*.json` (ID lists),
`results/frontier/`, `results/figures/`, `results/arm_table.tex`, and
`results/tags/{kstar,diagnostics,tagging_cost}.json`.

Gitignored because large and regenerable: `data/` (the 300 MB pool, MeSH,
MedExQA), `results/tags/tags.jsonl{,.gz}`, `results/tags/concepts.json`, and the
`overleaf*/` build directories and zips. Markdown working notes are also
untracked; `README.md` is the only tracked document.

---

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/smoke_test.py        # expect: ALL CHECKS PASSED
```

`tests/smoke_test.py` runs on a synthetic MeSH fragment and requires no data. It
verifies depth parsing, synonym collapse (*myocardial infarction* / *heart
attack* / *myocardial infarct* → one concept), ancestor propagation, saturation
measurement, and that every CPU selector respects its token budget.

The pipeline then runs in this order:

| Step | Script | Where | Notes |
|---|---|---|---|
| data | `prepare_data.py` | anywhere w/ internet | builds `pool.jsonl` + `medexqa.jsonl`, prints sha256s |
| gate 0 | `00_contamination.py` | GPU | zero-shot base accuracy. **Result: 0.423 — passed** (fails if > 0.60) |
| gate 1 | `01_tag.py` | local CPU | **Result: 23.04 tags/doc, 11.8% zero-tag — passed** (fails if < 3 tags/doc or ≥ 20% zero-tag) |
| gate 2 | `03_kstar.py` | local CPU | **Result: predicted k\*=575, observed exhaustion 3,523 — budgets bracket both** |
| select | `04_select.py` | local CPU | writes ID lists + per-arm cost records |
| train | `05_train.py` | GPU | fixed 500 steps, LoRA r=16, Qwen2.5-1.5B-Instruct |
| eval | `05b_eval.py` | GPU | writes `accuracy_{overall,head,tail,other}` into `metrics.json` |
| gate 3 | `06_placebo_gate.py` | anywhere | **Result: placebo ≈ ontology — FAILED, mechanism refuted** |
| analysis | `07_frontier.py`, `08_results_table.py`, `10_figures.py` | local | cost frontier, LaTeX table, all figures |
| paper | `09_write_results.py` | local | regenerates the paper's autogen block from run data |
| bundle | `11_bundle.py` | local | flat Overleaf zip |

### The one invariant that will silently invalidate every run

Selection outputs are **raw line indices into `pool.jsonl`** — `train.py` does
`corpus[i] for i in selected_ids`. If the selecting machine and the training
machine see different pool files, every arm is quietly garbage and no error is
raised. `pool.jsonl` is built **once**, sha256-hashed, and never regenerated; the
hash is recorded in every selection output. If it must be rebuilt, all tags and
selections die with it.

### The other invariants

- One shared selector signature; `train.py` never knows which arm it runs.
- `results/selections/` holds ID lists only, so runs reproduce without MeSH.
- `max_steps` fixed, **never epochs** — equal epochs over unequal sets gives
  unequal steps and an uninterpretable comparison.
- Every selector shuffles before returning; greedy order would leak a curriculum.
- Config hash in every `metrics.json` (all reported runs: `dcb6259f87b2`).
- Eval-set tagging goes through a code path **separate from selection**
  (stratification only).

---

## Compute

Selection, tagging, and all analysis are CPU-only and run on a laptop. Training
ran on **Beam** (`beam_app.py`), which superseded an earlier Kaggle-T4 plan and a
Modal attempt (`modal_app.py`).

```bash
beam cp data/pool/pool.jsonl onto-select://data/pool/pool.jsonl
beam cp results/selections onto-select://results/selections
python beam_app.py core --budget 1000000     # the 9 reported runs, in parallel
beam cp onto-select://results/runs results/runs
./status.sh watch                            # live gate + run dashboard
```

Each reported run is approximately 9 minutes of training. Because T4 is Turing,
the code uses `fp16` rather than `bf16` and `sdpa` rather than flash-attn-2 —
harmless on newer cards.

---

## Building the paper

`paper.latex` is the single source. Figures are referenced at their real paths
(`results/figures/*.png`, `results/tags/kstar_curve.png`), which Overleaf will
not resolve, hence the bundling step:

```bash
python scripts/10_figures.py        # regenerate all figures from results/
python scripts/09_write_results.py  # refresh the autogen results block
python scripts/11_bundle.py         # -> overleaf_flat/ + overleaf_flat.zip
```

`11_bundle.py` copies every `\includegraphics` target to the bundle root,
rewrites the paths, and zips it; `overleaf_flat.zip` uploads to Overleaf as-is.
It errors loudly if a referenced figure is missing. Both `overleaf_flat/` and the
zips are build outputs and are gitignored; regenerate them rather than editing
them.

`09_write_results.py` is idempotent: it rewrites everything between the
`% BEGIN AUTOGEN training-comparisons` and `% END AUTOGEN` markers in
`paper.latex` from whatever runs currently carry config hash `dcb6259f87b2`.
Nothing inside those markers should be hand-edited.

---

## Status

Complete and defensible as a registered report with a diagnosed null. Not
executed: the `d4` arm (repeated container SIGSEGVs on the available GPU tier —
reported analytically in FLOPs, not wall-clock), the full budget sweep, the five
ablations, and `02_validate_tagger.py` (tagger precision/recall against BioASQ).
