"""Beam app: GPU stages on H100. CPU stages (tagging, selection) run locally.

Setup:
    pip install beam-client
    beam configure default --token <token from dashboard.beam.cloud>

Upload local artifacts to the volume (syntax: beam --help / beam cp):
    beam cp data/pool/pool.jsonl onto-select://data/pool/pool.jsonl
    beam cp data/eval/medexqa.jsonl onto-select://data/eval/medexqa.jsonl
    beam cp results/tags onto-select://results/tags
    beam cp results/selections onto-select://results/selections

Run:
    python beam_app.py gate0
    python beam_app.py train --arm uniform --budget 500000 --seed 0
    python beam_app.py core --budget 500000        # 9 runs in parallel
    python beam_app.py d4 --budget 500000 --seed 0

Fetch results:
    beam cp onto-select://results/runs results/runs
"""
import json
import os
import subprocess
import sys

from beam import Image, Volume, function

GPU = os.environ.get("BEAM_GPU", "A10G")

VOL = Volume(name="onto-select", mount_path="/vol")
IMAGE = Image(python_version="python3.11").add_python_packages([
    "torch", "transformers>=4.44,<5", "peft<0.18", "accelerate", "datasets",
    "pyyaml", "pyahocorasick", "numpy", "scipy",
])


def _gunzip(path):
    import gzip, shutil
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rb") as f, open(path, "wb") as g:
            shutil.copyfileobj(f, g)


def _setup():
    os.environ["HF_HOME"] = "/vol/hf"   # cache model weights across runs
    for name in ("data", "results"):
        os.makedirs(f"/vol/{name}", exist_ok=True)
        if not os.path.islink(name):
            if os.path.isdir(name):
                os.rename(name, name + ".local")
            os.symlink(f"/vol/{name}", name)
    _gunzip("/vol/data/pool/pool.jsonl")        # uploads are gzipped to save
    _gunzip("/vol/results/tags/tags.jsonl")     # uplink bandwidth


def _run(cmd):
    subprocess.run(cmd, check=True)


@function(gpu=GPU, cpu=4, memory="16Gi", image=IMAGE, volumes=[VOL], timeout=1800)
def gate0(limit: int = 300):
    _setup()
    _run(["python", "scripts/00_contamination.py", "--limit", str(limit)])


@function(gpu=GPU, cpu=4, memory="16Gi", image=IMAGE, volumes=[VOL], timeout=7200)
def train_eval(spec: dict) -> dict:
    _setup()
    arm, budget, seed = spec["arm"], spec["budget"], spec["seed"]
    base = ["--arm", arm, "--budget", str(budget), "--seed", str(seed)]
    _run(["python", "scripts/05_train.py", *base])
    _run(["python", "scripts/05b_eval.py", *base])
    return json.load(open(f"results/runs/{arm}_{budget}_{seed}/metrics.json"))


@function(gpu=GPU, cpu=4, memory="16Gi", image=IMAGE, volumes=[VOL], timeout=3600)
def select_d4(budget: int, seed: int):
    """The one selector that needs a GPU; timing it here is H2's data point."""
    _setup()
    _run(["python", "-m", "pip", "install", "-q", "sentence-transformers<4", "scikit-learn"])
    _run(["python", "scripts/04_select.py", "--arm", "d4",
          "--budget", str(budget), "--seed", str(seed)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate0", "train", "core", "d4"])
    ap.add_argument("--arm")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="uniform,ontology,placebo")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    if a.cmd == "gate0":
        gate0.remote()
    elif a.cmd == "train":
        print(train_eval.remote({"arm": a.arm, "budget": a.budget, "seed": a.seed}))
    elif a.cmd == "d4":
        select_d4.remote(a.budget, a.seed)
    elif a.cmd == "core":
        specs = [{"arm": arm, "budget": a.budget, "seed": int(s)}
                 for arm in a.arms.split(",") for s in a.seeds.split(",")]
        for m in train_eval.map(specs):
            print(m["arm"], "seed", m["seed"],
                  "acc", m.get("accuracy_overall"), "tail", m.get("accuracy_tail"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
