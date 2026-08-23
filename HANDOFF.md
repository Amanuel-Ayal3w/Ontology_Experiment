# Handoff: Ontological Coverage Selection

**State as of 2026-08-24: the experiment is finished and the result is a
diagnosed null.** All four pre-registered gates were run; gate 3 (the mechanism
control) failed. The paper is written through the Conclusion with real numbers.
What remains is optional extension work, not completion work.

This document is for whoever picks the project up next — including future-you.
It records the corrections made along the way so they are not re-introduced,
and it is blunt about what was never executed.

Original context: four-week research assignment (Gheero, "Data and
Compute-Efficient Generative AI", July 2026), run from Addis Ababa on rented
GPU by the hour.

---

## 1. The claim, and what happened to it

The claim under test:

> All existing data selection methods require GPU compute to decide what to
> keep. This one does not. Here is what that costs in quality.

The answer: **it costs the entire effect.** Coverage selection using real MeSH
tags is indistinguishable from the same algorithm run on tags permuted across
examples. Whatever the selector retains, it is not ontological concept
coverage.

The paper's framing follows from this, and it is not an apology. The design
carried a placebo arm that this literature does not customarily include, and
that arm is what converts "our method didn't win" into "the mechanism does not
exist at this scale, and here is what does the work instead."

### What was never claimed

Each of these was considered and rejected during scoping. Do not re-introduce
them:

- Not better subsets than embedding-based selection. D4's signal is strictly
  richer; ontology tags see only what MeSH encodes and the matcher catches.
- Not a new training paradigm, architecture, or optimizer.
- Not a large absolute win. The ~4× FLOP reduction comes from *subsetting*,
  which uniform sampling also achieves.
- Not "we introduce cost-accounted data selection" — see §2, this was an error.
- Not applicable to domains without a curated ontology.

---

## 2. Corrections — do not re-introduce these

**Error 1: "nobody reports selection cost."** False. D4 (arXiv 2308.12284)
does the full accounting — 888 GPU-hours to embed 400B tokens against a gross
saving of 4,300, net 3,412 — and makes the CPU-vs-GPU price argument verbatim.
OFA, OST, and LENSLLM all report selection cost. The cost-accounted frontier is
established practice, not a contribution. The paper cedes it explicitly.

**Error 2: "continual pretraining relaxes the saturation bound."** Backwards.
`k* ≈ |C_eff|/m`, and full documents carry *more* concepts per unit (m ≈ 40 vs
≈ 5 for QA pairs), so saturation arrives at *fewer* documents. Passage chunking
is the fix; it lowers m and pushes k* out proportionally.

**Error 3: curriculum ordering as the primary lever.** Ordering buys maybe
1.1–1.3× and its literature is unreliable. Selection buys 2–4×. The project
moved from ordering to selection for this reason.

**Error 4 (added post-hoc): treating k\* as an exhaustion point.** It is not.
Measured: predicted 575, elbow in that region, but complete coverage only at
3,523 — roughly 6× out. `k*` marks where *binary* coverage stops
discriminating; the diminishing-returns objective stays informative well past
it. The paper says this in "What survives" and it is one of the two results
that outlive the null.

---

## 3. Method — as executed

Resource: **MeSH 2026**, categories C (Diseases) and D (Chemicals/Drugs) →
15,756 descriptors supplying 152,183 match terms after ambiguity filtering.

1. **Lexical grounding** — Aho–Corasick over labels + entry terms → concept IDs.
   Filters: drop terms < 4 chars, drop common English words. Discard rate 0.1%.
2. **Taxonomic positioning** — depth from tree-number segment count; ancestors
   by truncating segments. MeSH was chosen over SNOMED precisely because tree
   numbers encode depth directly, avoiding polyhierarchy ambiguity and a BFS.
3. **Coverage maximisation** — greedy submodular,
   `g(x) = (1/|x|) · Σ_{c∈φ(x)} w(c)/(1+n_c)`, then shuffle.

No reasoner, no OWL axioms, no triples, no inference.

Two design points that turned out to be load-bearing in an unexpected way:

- **Diminishing returns, not binary coverage.** Binary saturates at k*, past
  which selection degenerates to random. `w/(1+n_c)` keeps it informative.
- **Per-token normalisation.** Without `1/|x|` the objective just picks the
  longest documents. Since budget is denominated in tokens, gain-per-token is
  the correct quantity — **and this term is what survived the placebo.** It
  rewards short, tag-dense examples regardless of whether the tags mean
  anything. That is the discussion's central point.

**Keep this sentence in any methodology section:**

> The corpus is not transformed into an ontological representation. Concept
> recognition produces a stand-off annotation layer used solely for subset
> selection; the training data itself remains unmodified natural language.

It preempts a reviewer assuming semantic annotation / ontology population and
asking for extraction F1, which does not apply.

---

## 4. Configuration — as run

| | Choice | Notes |
|---|---|---|
| Model | Qwen2.5-1.5B-Instruct | large enough that seed variance doesn't drown the effect |
| Adapter | LoRA r=16, q/k/v/o | ~9 min/run |
| Steps | `max_steps: 500`, batch 2 × grad-accum 8 | **fixed across arms, never epochs** |
| Pool | MedS-Ins, 167,102 examples over 62 English tasks, ≤4,000/task | mean ~442 tokens |
| Eval | MedExQA, 940 test items across five underrepresented specialties | |
| Budget reported | 1M tokens (~2,262 examples uniform, 1.4% keep ratio) | below exhaustion |
| Seeds | 0, 1, 2 per arm | three is the floor |
| Hardware | Beam (`beam_app.py`) | Kaggle plan abandoned; Modal attempt superseded |
| Config hash | `dcb6259f87b2` | present in every reported `metrics.json` |

Aggressive keep ratios matter: at 25% retention all methods pick reasonable
data and the arms converge. 1.4% is well inside the regime where selection
should have mattered.

---

## 5. Hypotheses — resolved

- **H1** — tail accuracy above uniform below k\*. **Not supported.** +0.016
  (p = 0.40); per-seed +0.0117 / −0.0079 / +0.0431, inconsistent in sign.
- **H2** — selection compute ≥2 orders below D4. **Supported in FLOPs, not
  wall-clock.** 42 CPU-s and 0 GPU-s against 3.35e15 encoder FLOPs. No measured
  D4 timing (see §7).
- **H3** — Pareto-nondominated. **Fails**, as pre-specified: uniform matches
  ontology within seed variance, and uniform is also free.
- **H4** — advantage diminishes above k\*. **Untested** — only the 1M budget
  was run. The sweep is the largest missing piece.
- **H5** — gains concentrate on tail, head not degraded. **Not supported**; no
  stratum separates the arms beyond seed spread.
- **Mechanism** — fails if placebo ≈ ontology. **It does.** −0.007 (p = 0.71).

The null is diagnostic because the design excludes the two mundane
explanations before they can be offered:

- *Not* operating above saturation: 1M tokens selects ~2,262 examples against
  observed exhaustion at 3,523, so the objective was still discriminating.
- *Not* insufficient tagging recall: 6.16 direct concepts per example (23.04
  with ancestors), only 11.8% zero-tag, 0.1% term discard. The tagger saw the
  corpus.

What remains is the possibility flagged as most likely from the start: coverage
is a proxy for learning value, and a passing mention of a rare concept does not
teach that concept.

---

## 6. Codebase state

### Executed and trusted

The whole CPU path — `src/ontology/{mesh,tagger}.py`,
`src/selection/{coverage,baselines,registry}.py`, `src/cost.py` — ran over the
real 167k pool and real MeSH `d2026.bin`, not just the synthetic fragment.
`tests/smoke_test.py` still passes (it is a script, not pytest-discoverable) and
remains the fastest sanity check after any edit.

`src/train.py` and `src/evaluate.py` were written from knowledge and *have* now
been executed — nine training runs and nine evaluations. The predicted failure
points (`format_example()` field names, tokenizer padding, dtype) were real and
are fixed.

### Never executed

- `baselines.d4_select` — the real competitor. Repeated native crashes
  (SIGSEGV) in the container on the available GPU tier. The paper reports the
  encoder cost analytically as `2P'T` and says so explicitly rather than
  splicing in a timing from different hardware, which the metrics section
  forbids.
- `scripts/02_validate_tagger.py` — **not written.** Tagger P/R against
  BioASQ-QA human concept annotations. Deferred pending BioASQ's actual
  concept-field schema and registration. The paper commits to this measurement;
  right now the tagging quality argument rests on the diagnostics
  (density, zero-tag rate, discard rate), which is weaker.

### Invariants — do not break when editing

- Every selector shares one signature; `train.py` never knows which arm it runs.
- `results/selections/` holds ID lists only, so runs reproduce without MeSH.
- Those IDs are **raw line indices into `pool.jsonl`**. Different pool file =
  silently garbage arms, no error. Pool sha256 is recorded in every selection.
- `max_steps` fixed, never epochs.
- Every selector shuffles before returning.
- Config hash in every `metrics.json`.
- Eval-set tagging runs through a code path separate from selection.

---

## 7. Gates — all run

| # | Script | Threshold | Measured | Verdict |
|---|---|---|---|---|
| 0 | `00_contamination.py` | fail if base > 0.60 | 0.423 | **pass** |
| 1 | `01_tag.py` | fail if < 3 tags/doc or ≥ 20% zero-tag | 23.04 tags/doc, 11.8% zero-tag | **pass** |
| 2 | `03_kstar.py` | fail if k\* far outside budget ladder | predicted 575, exhaustion 3,523; ladder brackets both | **pass** |
| 3 | `06_placebo_gate.py` | fail if placebo ≈ ontology | −0.007, p = 0.71 | **FAIL — mechanism refuted** |

Gate 3 was deliberately pulled forward rather than left to week 4. That
decision is why there was time to write the null up properly instead of
discovering it at the deadline.

---

## 8. Reading

- **D4** (arXiv 2308.12284) — closest methodological neighbour; §4.3 is the cost
  analysis. Form your own view on the delta; assistant summaries proved
  unreliable here once already.
- **arXiv 2503.22006** — "Enhancing Domain-Specific Encoder Models with
  LLM-Generated Data: How to Leverage Ontologies, and How to Do Without Them."
  The sharpest challenge to the project, and this result lands on its side.
- OntoTune (arXiv 2502.05478), code at github.com/zjukg/OntoTune
- Evontree (arXiv 2510.26683)

Citation hygiene: several arXiv IDs in `paper.latex` were drafted from search
snippets. Verify every one resolves before submission — a fabricated citation
in a registered report is fatal. `\bibitem{ost}` in particular carried a
placeholder title.

---

## 9. If the project continues

In descending order of value:

1. **The budget sweep** (250k / 500k / 2.5M). H4 is the one hypothesis with no
   evidence either way, and the curve is what would show whether the ontology
   ever mattered at *some* budget. ~27 more runs.
2. **`02_validate_tagger.py`.** Cheap, CPU-only, and it closes the one
   measurement the paper promises and does not deliver.
3. **A working `d4` arm** on a GPU tier that doesn't crash — converts H2 from a
   FLOP argument into a measured one.
4. **The ablations — but reconsider them.** Flat-vs-propagated, weighting, and
   saturation ablations all assume the concept signal contributes something.
   The placebo says it does not. The prior question is whether concept signal
   survives a permuted control *at any budget*, and it should be asked first.
5. **Ablation 5 (cost scaling)** is the exception: CPU-only, needs no ontology
   premise, and directly strengthens the surviving cost result by showing the
   ratio grows with pool size while training cost does not.

### The fallback that is still on the table

**Ontology-guided synthetic generation.** Rather than selecting from a corpus,
walk the ontology and generate examples for uncovered concepts. It is stronger
than selection in four ways: the delta is the whole effect (no free baseline
gets coverage), it applies where no corpus exists, it sidesteps saturation, and
it runs concept → text rather than text → concept — so it needs no
morphological analyser, which is exactly what makes Amharic *selection*
infeasible and Amharic *generation* not. Metric: % fewer generation calls and
tokens for equal concept coverage; expect 40–60%, since unguided generation is
heavily Zipfian.

Note that this project's null does not damage that one. The refuted claim is
that *coverage predicts learning value when selecting existing text*. Generation
does not depend on it.
