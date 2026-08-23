# Kaggle Notebook: Ontological Coverage Selection

> **Historical — superseded.** This was the original plan when the project
> targeted Kaggle T4s. Training actually ran on Beam (`beam_app.py`); session
> kills and queue latency made Kaggle unworkable for the run matrix. Kept
> because the cell-by-cell gate ordering and the session-survival notes are
> still the clearest description of how the stages fit together. Nothing here
> reflects the final configuration — see `README.md`.

Each `python` block below is one notebook cell. Settings: **Accelerator → GPU T4 x2**, **Internet → On**.

T4 is Turing, so: `fp16` not `bf16`, and `sdpa` attention not flash-attn-2. Both are already set in the code.

---

## Cell 1 — Setup

```python
!pip install -q pyahocorasick peft accelerate sentence-transformers

import os, sys, json, time, random
from pathlib import Path
import numpy as np, torch

WORK = Path("/kaggle/working/onto-select")
sys.path.insert(0, str(WORK))
print("GPU:", torch.cuda.get_device_name(0))
```

Upload the codebase as a Kaggle Dataset, then:

```python
!cp -r /kaggle/input/onto-select-code/* /kaggle/working/
!ls /kaggle/working/onto-select
```

---

## Cell 2 — Verify the code works

Run this before anything else. It uses a synthetic MeSH fragment, needs no data, and takes two seconds.

```python
!cd /kaggle/working/onto-select && python tests/smoke_test.py
```

Expect `ALL CHECKS PASSED`, and note the line showing that *myocardial infarction*, *heart attack*, and *myocardial infarct* map to one concept. That is the mechanism in miniature — three documents sharing no vocabulary, detected as redundant.

---

## Cell 3 — GATE 0: contamination check

**Do not skip this.** If the base model already scores well, fine-tuning has no headroom and every arm converges regardless of selection quality.

```python
# MedExQA ships as per-specialty TSVs (no HF loader). 940 test items.
!cd /kaggle/working/onto-select && python scripts/prepare_data.py --skip-pool
```

```python
!cd /kaggle/working/onto-select && python scripts/00_contamination.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --eval data/eval/medexqa.jsonl --limit 300
```

| Result | Meaning |
|---|---|
| accuracy 0.30–0.55 | Good headroom — proceed |
| accuracy > 0.60 | Ceiling risk — switch to MedXpertQA (10-option) |
| accuracy ≈ chance | Model too small or prompt format wrong |

---

## Cell 4 — Get MeSH and the candidate pool

MeSH needs a free NLM account. Download `d2026.bin` from the [ASCII MeSH archive](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/) and upload it as a Kaggle Dataset.

```python
!mkdir -p /kaggle/working/onto-select/data/mesh
!cp /kaggle/input/mesh-2026/d2026.bin /kaggle/working/onto-select/data/mesh/
!ls -lh /kaggle/working/onto-select/data/mesh/
```

Candidate pool — sample 200k from MedS-Ins:

```python
# MedS-Ins is a repo of per-task JSON files, NOT load_dataset()-able.
# prepare_data.py samples per task (English only, files <400MB skipped)
# and prints the pool sha256 -- record it in the Kaggle Dataset description.
!cd /kaggle/working/onto-select && python scripts/prepare_data.py --skip-eval
```

Save both as a Kaggle Dataset now so you aren't re-downloading every session.

---

## Cell 5 — GATE 1: tag and check density

```python
!cd /kaggle/working/onto-select && python scripts/01_tag.py \
    --mesh data/mesh/d2026.bin --pool data/pool/pool.jsonl --categories C,D
```

Two gates must pass:

- `mean_tags_per_doc ≥ 3` — below this there is too little signal for coverage to exploit
- `zero_tag_rate < 0.20` — above this the tagger is too strict

If density is low, loosen `min_term_len` to 3 or widen categories to `C,D,E`. If the zero-tag rate is high, your corpus may not be concept-dense — weight toward longer-form sources.

Also note `k_star_predicted` in the output. That number determines your budget ladder.

---

## Cell 6 — GATE 2: measure saturation (Proof 1)

No GPU needed. **This is a reportable result on its own.**

```python
!cd /kaggle/working/onto-select && python scripts/03_kstar.py --max-steps 8000
```

```python
from IPython.display import Image
Image("/kaggle/working/onto-select/results/tags/kstar_curve.png")
```

The elbow in the marginal-gain curve is your observed $k^*$. Compare it to the predicted $|C_{\text{eff}}|/m$ — agreement validates the bound, disagreement is itself a finding.

**Set your budgets to bracket $k^*$.** Selecting far above it means coverage has gone inert and your method is doing random selection in disguise.

---

## Cell 7 — Run the selectors

```python
K_STAR = json.load(open("/kaggle/working/onto-select/results/tags/kstar.json"))["k_star_observed"]
mean_len = 250   # approx tokens per example; check your own pool
BUDGET = int(K_STAR * mean_len * 0.75)   # comfortably below saturation
print("budget tokens:", BUDGET)
```

```python
for arm in ["uniform", "ontology", "placebo"]:
    for seed in [0, 1, 2]:
        !cd /kaggle/working/onto-select && python scripts/04_select.py \
            --arm {arm} --budget {BUDGET} --seed {seed}
```

D4 separately — this is the arm that costs GPU time, and timing it is the point:

```python
for seed in [0, 1, 2]:
    !cd /kaggle/working/onto-select && python scripts/04_select.py \
        --arm d4 --budget {BUDGET} --seed {seed}
```

```python
# Proof 2: the cost comparison
import glob
for p in sorted(glob.glob("/kaggle/working/onto-select/results/selections/*_cost.json")):
    print(Path(p).stem, json.load(open(p)))
```

Report the **ratio**, not the absolute saving. At 200k examples D4's encoding is minutes, not hours — the argument is that the ratio scales linearly with corpus size while training cost does not.

---

## Cell 8 — Train

~20–30 min per run on T4. **Save after every run** — Kaggle sessions die.

```python
for arm in ["uniform", "ontology", "placebo"]:
    for seed in [0, 1, 2]:
        print(f"=== {arm} seed {seed} ===")
        !cd /kaggle/working/onto-select && python scripts/05_train.py \
            --arm {arm} --budget {BUDGET} --seed {seed}
```

Nine runs ≈ 4 hours. Split across sessions if needed; results are written per-run so partial progress survives.

---

## Cell 9 — GATE 3: the placebo check

**Run this before training any further arms.**

```python
!cd /kaggle/working/onto-select && python scripts/06_placebo_gate.py --metric accuracy_tail
```

| Outcome | What it means | What to do |
|---|---|---|
| ontology > uniform **and** ontology > placebo | Mechanism supported | Scale up: D4, budget sweep |
| ontology ≈ placebo | Gain is from selection *shape*, not concepts | Stop. Rethink. |
| ontology ≈ uniform | Null result | Report it — the design makes it interpretable |

A null here is not a failed project. With the budget curve, the placebo arm, and the cost measurement, you can say *why* it was null — which most negative results in this literature cannot.

---

## Cell 10 — Results table

```python
import sys; sys.path.insert(0, "/kaggle/working/onto-select")
from src.evaluate import aggregate_runs, paired_test

print(json.dumps(aggregate_runs("/kaggle/working/onto-select/results/runs"), indent=2))
for a, b in [("ontology","uniform"), ("ontology","placebo"), ("ontology","d4")]:
    print(json.dumps(paired_test("/kaggle/working/onto-select/results/runs", a, b), indent=2))
```

The table your paper is built around:

```
Arm        Sel GPU-hr  Sel wall-clock  Acc (mean±std)  Tail Acc
──────────────────────────────────────────────────────────────
Uniform         0          ~0            ..  ± ..        ..
Ontology        0          ..            ..  ± ..        ..
D4             ..          ..            ..  ± ..        ..
Placebo         0          ..            ..  ± ..        ..
```

---

## Session survival

Kaggle kills sessions at 9 hours and on disconnect. Practical measures:

- After each batch, `!zip -r /kaggle/working/results.zip /kaggle/working/onto-select/results` and download
- Save the tagged corpus as a Kaggle Dataset — re-tagging 200k examples each session wastes real time
- Run one arm per session rather than queuing all nine

---

## Week plan

| Week | Cells | Gate |
|---|---|---|
| 1 | 1–6 | Contamination, density, $k^*$ |
| 2 | 7–9 | Placebo |
| 3 | D4 arm, budget sweep | — |
| 4 | Ablations, writeup | — |

Cells 1–7 produce reportable results (tagger validation, $k^*$, cost comparison) regardless of what happens in training. That is your safety margin — two solid sections before you know whether the third works.
