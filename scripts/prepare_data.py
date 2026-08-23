#!/usr/bin/env python3
"""Build pool.jsonl (MedS-Ins) and eval jsonl (MedExQA).

MedS-Ins is NOT a load_dataset()-able dataset: it is a repo of
Super-NaturalInstructions-style task files, each
{Definition: [...], Instances: [{id, input, output}, ...], ...}.
So we list the repo, sample per task, and flatten to
{instruction, input, output, task}. Run where bandwidth is good (Kaggle).
"""
import argparse, csv, hashlib, json, random, sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

POOL_REPO = "Henrychur/MedS-Ins"
EVAL_REPO = "bluesky333/MedExQA"
SPECIALTIES = [
    "biomedical_engineer",
    "clinical_laboratory_scientist",
    "clinical_psychologist",
    "occupational_therapist",
    "speech_pathologist",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pool(out: Path, target: int, cap: int, max_file_mb: int, seed: int) -> None:
    rng = random.Random(seed)
    api = HfApi()
    files = sorted(
        (e.path, e.size)
        for e in api.list_repo_tree(POOL_REPO, repo_type="dataset")
        if e.path.startswith("task") and e.path.endswith(".json")
    )
    records, skipped = [], []
    for name, size in files:
        if size > max_file_mb * 1024 * 1024:
            skipped.append((name, "size"))
            continue
        try:
            local = hf_hub_download(POOL_REPO, name, repo_type="dataset")
            task = json.load(open(local, encoding="utf-8"))
        except Exception as e:
            skipped.append((name, f"error: {e}"))
            continue
        if task.get("Input_language") != ["English"] or task.get("Output_language") != ["English"]:
            skipped.append((name, "non-english"))
            continue
        instruction = (task.get("Definition") or [""])[0]
        inst = task.get("Instances") or []
        for r in rng.sample(inst, min(cap, len(inst))):
            output = r.get("output", "")
            if isinstance(output, list):
                output = output[0] if output else ""
            records.append({
                "instruction": instruction,
                "input": r.get("input", ""),
                "output": str(output),
                "task": name,
            })
        print(f"{name}: kept {min(cap, len(inst))}/{len(inst)}")
    if len(records) > target:
        records = rng.sample(records, target)
    rng.shuffle(records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tasks = {r["task"] for r in records}
    print(f"\npool: {len(records)} examples from {len(tasks)} tasks -> {out}")
    print(f"pool sha256: {sha256_of(out)}")
    for name, why in skipped:
        print(f"  skipped {name} ({why})")


def build_eval(out: Path, split: str) -> None:
    items = []
    for spec in SPECIALTIES:
        local = hf_hub_download(EVAL_REPO, f"{split}/{spec}_{split}.tsv", repo_type="dataset")
        with open(local, encoding="utf-8") as f:
            # no header; cols: question, A, B, C, D, expl1, expl2, answer letter
            for i, row in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
                if len(row) != 8 or row[7].strip() not in "ABCD":
                    print(f"  {spec} row {i}: malformed, skipped")
                    continue
                items.append({
                    "question": row[0],
                    "options": row[1:5],
                    "answer_idx": "ABCD".index(row[7].strip()),
                    "specialty": spec,
                })
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"eval: {len(items)} items -> {out}")
    print(f"eval sha256: {sha256_of(out)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-out", default="data/pool/pool.jsonl")
    ap.add_argument("--eval-out", default="data/eval/medexqa.jsonl")
    ap.add_argument("--target", type=int, default=200_000)
    ap.add_argument("--cap", type=int, default=4000, help="max instances per task")
    ap.add_argument("--max-file-mb", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-split", default="test", choices=["test", "dev"])
    ap.add_argument("--skip-pool", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    if not a.skip_eval:
        build_eval(Path(a.eval_out), a.eval_split)
    if not a.skip_pool:
        build_pool(Path(a.pool_out), a.target, a.cap, a.max_file_mb, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
