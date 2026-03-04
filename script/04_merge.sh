#!/usr/bin/env bash
# Merge LoRA adapter into base SparkTTS LLM and save a standalone model checkpoint.
# The merged checkpoint can be used with run_inference.py (no --adapter_dir needed).

CUDA_VISIBLE_DEVICES=7 uv run python finetune/merge_lora.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --adapter_dir checkpoints/vieneu-lora \
  --output_dir checkpoints/vieneu-merged \
  --dtype bfloat16
