"""
Training harness.

Deliberately knows NOTHING about selection. It consumes an ID list and a
seed. This is what guarantees every arm runs byte-identical training code,
so any difference is attributable to set membership alone.

Two things that silently break matched comparisons and are enforced here:
  * max_steps is FIXED, not epochs. 500 examples x 3 epochs is 1500 steps;
    5000 x 3 is 15000. Comparing those measures step count, not data quality.
  * The config hash is written into metrics.json, so 'the arms were matched'
    is demonstrable rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_hash(cfg: dict) -> str:
    """Hash of everything except the arm name and seed."""
    stable = {k: v for k, v in sorted(cfg.items()) if k not in ("arm", "seed")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:12]


def format_example(rec: dict) -> str:
    """Adapt to your corpus schema. MedS-Ins-style instruction format."""
    instr = rec.get("instruction", "").strip()
    inp = rec.get("input", "").strip()
    out = rec.get("output", "").strip()
    prompt = f"{instr}\n\n{inp}".strip() if inp else instr
    return f"<|user|>\n{prompt}\n<|assistant|>\n{out}"


def train_arm(
    corpus: list[dict],
    selected_ids: list[int],
    cfg: dict,
    seed: int,
    out_dir: str | Path,
) -> dict:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)

    texts = [format_example(corpus[i]) for i in selected_ids]
    random.Random(seed).shuffle(texts)  # kill any greedy ordering

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=cfg["max_seq_len"],
            padding=False,
        )

    ds = Dataset.from_dict({"text": texts}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_targets"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        max_steps=cfg["max_steps"],          # FIXED across arms
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=25,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    t0 = time.perf_counter()
    result = trainer.train()
    train_seconds = time.perf_counter() - t0

    model.save_pretrained(out_dir / "adapter")

    n_tokens = sum(len(x["input_ids"]) for x in ds)
    metrics = {
        "arm": cfg["arm"],
        "seed": seed,
        "config_hash": config_hash(cfg),
        "n_examples": len(selected_ids),
        "n_train_tokens": n_tokens,
        "max_steps": cfg["max_steps"],
        "final_loss": result.training_loss,
        "train_seconds": round(train_seconds, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
