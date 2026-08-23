"""Modal app: GPU stages on H100. CPU stages (tagging, selection) run locally.

One-time setup:
    pip install modal && modal setup

Upload local artifacts to the volume:
    modal volume put onto-select data/pool/pool.jsonl data/pool/pool.jsonl
    modal volume put onto-select data/eval/medexqa.jsonl data/eval/medexqa.jsonl
    modal volume put onto-select results/tags results/tags
    modal volume put onto-select results/selections results/selections

Run:
    modal run modal_app.py::gate0
    modal run modal_app.py::core --budget 500000        # 9 runs in parallel
    modal run modal_app.py::train_one --arm d4 --budget 500000 --seed 0

Fetch results:
    modal volume get onto-select results/runs results/runs
"""
import subprocess
import modal

app = modal.App("onto-select")
vol = modal.Volume.from_name("onto-select", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers>=4.44", "peft", "accelerate", "datasets",
        "pyyaml", "pyahocorasick", "numpy", "scipy",
    )
    .add_local_dir(
        ".", remote_path="/root/onto-select",
        ignore=[".git", ".venv", "data", "results", "__pycache__", "*.pyc",
                "notebooks", ".claude"],
    )
)


def _setup():
    import os
    os.environ["HF_HOME"] = "/vol/hf"   # cache model weights across runs
    for name in ("data", "results"):
        os.makedirs(f"/vol/{name}", exist_ok=True)
        dst = f"/root/onto-select/{name}"
        if not os.path.islink(dst):
            os.symlink(f"/vol/{name}", dst)


def _run(cmd):
    subprocess.run(cmd, cwd="/root/onto-select", check=True)


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=1800)
def gate0(limit: int = 300):
    _setup()
    _run(["python", "scripts/00_contamination.py", "--limit", str(limit)])
    vol.commit()


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=7200)
def train_eval(arm: str, budget: int, seed: int) -> dict:
    _setup()
    base = ["--arm", arm, "--budget", str(budget), "--seed", str(seed)]
    _run(["python", "scripts/05_train.py", *base])
    _run(["python", "scripts/05b_eval.py", *base])
    vol.commit()
    import json, pathlib
    return json.loads(pathlib.Path(
        f"/vol/results/runs/{arm}_{budget}_{seed}/metrics.json").read_text())


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=3600)
def select_d4(budget: int, seed: int):
    """The one selector that needs a GPU; timing it here is H2's data point."""
    _setup()
    _run(["python", "scripts/04_select.py", "--arm", "d4",
          "--budget", str(budget), "--seed", str(seed)])
    vol.commit()


@app.local_entrypoint()
def train_one(arm: str, budget: int, seed: int = 0):
    print(train_eval.remote(arm, budget, seed))


@app.local_entrypoint()
def core(budget: int, arms: str = "uniform,ontology,placebo", seeds: str = "0,1,2"):
    combos = [(a, budget, int(s))
              for a in arms.split(",") for s in seeds.split(",")]
    for m in train_eval.starmap(combos):
        print(m["arm"], "seed", m["seed"],
              "acc", m.get("accuracy_overall"), "tail", m.get("accuracy_tail"))
