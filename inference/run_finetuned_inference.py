#!/usr/bin/env python3
"""
Run SparkTTS inference with a finetuned LoRA checkpoint.

Example:
  uv run python inferece/run_finetuned_inference.py \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --adapter_dir checkpoints/ngochuyen-demo \
    --text "Xin chao, day la ban thu nghiem voi checkpoint moi." \
    --prompt_speech_path LJSpeech-1.1/wavs/LJ050-0001.wav \
    --output_path outputs/output_finetuned.wav
"""
import sys
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] + "/..")


import argparse
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cli.SparkTTS import SparkTTS


def build_normalizer(name: str):
    if name == "none":
        return None
    if name == "vinorm":
        try:
            from vinorm import TTSnorm
        except ImportError as exc:
            raise ImportError(
                "vinorm is not installed. Run: uv pip install vinorm"
            ) from exc
        return lambda text: TTSnorm(text)
    raise ValueError(f"Unknown text_normalizer: {name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using a SparkTTS base model + LoRA adapter checkpoint."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("pretrained_models/Spark-TTS-0.5B"),
        help="SparkTTS base model directory (contains LLM/ and config.yaml).",
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        required=True,
        help="LoRA adapter checkpoint directory (e.g. checkpoints/ngochuyen-demo).",
    )
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument(
        "--prompt_speech_path",
        type=Path,
        default=None,
        help="Prompt wav path for voice cloning mode.",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Transcript of prompt audio (optional).",
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default=None,
        help="Use controllable mode when provided.",
    )
    parser.add_argument(
        "--pitch",
        choices=["very_low", "low", "moderate", "high", "very_high"],
        default="moderate",
        help="Pitch control for controllable mode.",
    )
    parser.add_argument(
        "--speed",
        choices=["very_low", "low", "moderate", "high", "very_high"],
        default="moderate",
        help="Speed control for controllable mode.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--text_normalizer",
        choices=["none", "vinorm"],
        default="none",
        help=(
            "Text normalizer for --text. "
            "'vinorm' converts Vietnamese numbers/dates/abbreviations to spoken form "
            "(requires: uv pip install vinorm)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device string, e.g. "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/output_finetuned.wav"),
        help="Path to save generated wav.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def choose_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {args.model_dir}")
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"adapter_dir not found: {args.adapter_dir}")
    if args.gender is None and args.prompt_speech_path is None:
        raise ValueError(
            "Provide --prompt_speech_path for voice cloning mode, or set --gender for controllable mode."
        )
    if args.prompt_speech_path is not None and not args.prompt_speech_path.exists():
        raise FileNotFoundError(f"prompt_speech_path not found: {args.prompt_speech_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    normalizer = build_normalizer(args.text_normalizer)

    device = resolve_device(args.device)
    dtype = choose_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir / "LLM"),
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir)).to(device)
    model.eval()

    tts = SparkTTS(args.model_dir, device)
    tts.model = model
    tts.tokenizer = tokenizer

    text = normalizer(args.text) if normalizer else args.text
    with torch.no_grad():
        wav = tts.inference(
            text=text,
            prompt_speech_path=args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch if args.gender else None,
            speed=args.speed if args.gender else None,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output_path), wav, tts.sample_rate)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
