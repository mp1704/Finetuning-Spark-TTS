"""
Finetune the SparkTTS LLM on LJSpeech using LoRA.

BiCodec and Wav2Vec2 are kept frozen. Only the Qwen2.5 LLM receives LoRA adapters.
WandB API key is loaded from .env (WANDB_API_KEY=...).

Usage:
    uv run python -m finetune.train --config finetune/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env before any wandb / HF imports so the key is in the environment
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.distributed.tensor  # noqa: F401 — pre-import so peft can access torch.distributed.tensor.DTensor
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune.dataset import LJSpeechTTSDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="finetune/config.yaml",
        help="Path to training config YAML",
    )
    return parser.parse_args()


def build_model_and_tokenizer(cfg):
    llm_dir = Path(cfg.model_dir) / "LLM"
    print(f"Loading tokenizer from {llm_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(llm_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {llm_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(llm_dir),
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16 if cfg.training.fp16 else torch.float32,
    )

    # Required when gradient_checkpointing=True: PEFT detaches embeddings from
    # the graph, so we must re-attach them manually before wrapping with LoRA.
    model.enable_input_require_grads()

    lora_cfg = cfg.lora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_training_args(cfg) -> TrainingArguments:
    t = cfg.training
    wandb_cfg = cfg.get("wandb", {})

    # Pass wandb run name via env var (HF Trainer picks it up automatically)
    if wandb_cfg.get("run_name"):
        os.environ.setdefault("WANDB_RUN_NAME", wandb_cfg.run_name)
    if wandb_cfg.get("project"):
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.project)

    return TrainingArguments(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=t.warmup_steps,
        logging_steps=t.logging_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        bf16=t.bf16,
        fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing,
        dataloader_num_workers=t.dataloader_num_workers,
        remove_unused_columns=t.get("remove_unused_columns", False),
        optim=t.get("optim", "adamw_torch"),
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    model, tokenizer = build_model_and_tokenizer(cfg)

    train_dataset = LJSpeechTTSDataset(
        jsonl_path=cfg.data.train_jsonl,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )
    val_dataset = LJSpeechTTSDataset(
        jsonl_path=cfg.data.val_jsonl,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    training_args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    resume_ckpt = cfg.training.get("resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save the LoRA adapter weights
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
