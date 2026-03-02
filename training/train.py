"""
Fine-tune SparkTTS LLM on Vietnamese TTS data using LoRA.

Only the LLM (Qwen2.5-0.5B) is fine-tuned; BiCodec is not loaded during training
(tokens are pre-computed). WandB is used for experiment tracking.

Usage:
    python training/train.py --config training/configs/train_phase1.yaml
    python training/train.py --config training/configs/train_phase1.yaml \
        --resume_from checkpoints/phase1/checkpoint-2500
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.data.dataset import SparkTTSDataset, collate_fn

load_dotenv()  # reads WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY from .env


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    model_dir = cfg["model_dir"]
    llm_dir = f"{model_dir}/LLM"

    print(f"Loading tokenizer from {llm_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_dir)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {llm_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        llm_dir,
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    train_cfg = cfg["training"]
    if train_cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=train_cfg.get("lora_r", 16),
            lora_alpha=train_cfg.get("lora_alpha", 32),
            lora_dropout=train_cfg.get("lora_dropout", 0.05),
            target_modules=train_cfg.get(
                "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def build_datasets(cfg: dict, tokenizer):
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})

    metadata_path = data_cfg["metadata"]
    # Accept both plain metadata.jsonl and pre-tokenized version
    if not metadata_path.endswith("_tokenized.jsonl"):
        tok_path = metadata_path.replace(".jsonl", "_tokenized.jsonl")
        if Path(tok_path).exists():
            metadata_path = tok_path

    dataset = SparkTTSDataset(
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        max_seq_len=data_cfg.get("max_seq_len", 2048),
        normalize=data_cfg.get("normalize", False),
    )

    test_frac = eval_cfg.get("test_split", 0.05)
    n_test = max(1, int(len(dataset) * test_frac))
    n_train = len(dataset) - n_test
    train_ds, eval_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    return train_ds, eval_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume_from", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    output_dir = train_cfg.get("output_dir", "checkpoints/run")

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds, eval_ds = build_datasets(cfg, tokenizer)

    wandb_project = os.environ.get("WANDB_PROJECT", "sparktts-vi")
    wandb_entity = os.environ.get("WANDB_ENTITY", None)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        max_steps=train_cfg.get("max_steps", 5000),
        bf16=train_cfg.get("bf16", True),
        logging_steps=10,
        save_steps=train_cfg.get("save_steps", 500),
        eval_steps=train_cfg.get("eval_steps", 500),
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=cfg.get("run_name", Path(args.config).stem),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Set WandB env vars from .env if not already set
    if wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", wandb_entity)
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
