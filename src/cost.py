"""
Selection cost measurement (Proof 2).

Both selectors MUST be timed on the same corpus and the same machine.
Citing D4's published 888 GPU-hours against your own timing compares
different corpora on different hardware and is not a comparison.

Report wall-clock AND cost separately: CPU-hours are cheap but may be slow,
GPU-hours are fast but expensive. They tell different stories.
"""
import json, os, time
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def measure(name: str, sink: dict):
    t0 = time.perf_counter()
    c0 = time.process_time()
    yield
    sink[name] = {
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "cpu_seconds": round(time.process_time() - c0, 3),
        "n_cores": os.cpu_count(),
    }

def encoder_flops(n_params: int, n_tokens: int) -> float:
    """Forward-pass FLOPs ~= 2 * P * T."""
    return 2.0 * n_params * n_tokens

def training_flops(n_params: int, n_tokens: int) -> float:
    """Fwd+bwd FLOPs ~= 6 * P * T."""
    return 6.0 * n_params * n_tokens

def save(costs: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(costs, indent=2))
