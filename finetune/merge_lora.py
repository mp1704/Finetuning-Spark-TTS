"""
Merge a LoRA adapter into the base SparkTTS LLM and save a full model checkpoint.

The merged checkpoint preserves the SparkTTS directory layout:
  <output_dir>/
    LLM/          ← merged weights (full, no adapter)
    BiCodec/      ← symlinked from base model
    wav2vec2-large-xlsr-53/ ← symlinked from base model
    src/          ← symlinked from base model
    config.yaml   ← copied from base model

Usage:
    uv run python finetune/merge_lora.py \
        --model_dir pretrained_models/Spark-TTS-0.5B \
        --adapter_dir checkpoints/vieneu-lora \
        --output_dir checkpoints/vieneu-merged
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.tensor  # noqa: F401 — pre-import so peft can access torch.distributed.tensor.DTensor
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Spark-TTS-0.5B",
                        help="Path to the base SparkTTS model directory")
    parser.add_argument("--adapter_dir", type=str, default="checkpoints/vieneu-lora",
                        help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/vieneu-merged",
                        help="Where to save the merged SparkTTS model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Torch dtype for loading/saving the model")
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    adapter_dir = Path(args.adapter_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    llm_dir = model_dir / "LLM"
    out_llm_dir = output_dir / "LLM"

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    print(f"Base LLM  : {llm_dir}")
    print(f"Adapter   : {adapter_dir}")
    print(f"Output    : {output_dir}")
    print(f"Dtype     : {args.dtype}")

    # 1. Load base LLM
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(str(llm_dir), torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))

    # 2. Load LoRA adapter and merge
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_dir), torch_dtype=torch_dtype)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # 3. Save merged LLM
    out_llm_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged LLM to {out_llm_dir} ...")
    model.save_pretrained(str(out_llm_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_llm_dir))

    # 4. Symlink the other SparkTTS components from the base model
    symlink_targets = ["BiCodec", "wav2vec2-large-xlsr-53", "src"]
    for name in symlink_targets:
        src = model_dir / name
        dst = output_dir / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
        os.symlink(src, dst)
        print(f"Symlinked {name} -> {src}")

    # Copy config.yaml
    for fname in ["config.yaml", "README.md"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")

    print(f"\nDone. Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
