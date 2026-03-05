"""
Download the Zipformer-30M-RNNT-6000h Vietnamese ASR model from HuggingFace.

Usage:
    uv run python -m asr.download_model
    uv run python -m asr.download_model --model-dir asr/models/zipformer-30m
    uv run python -m asr.download_model --full    # also download full-precision ONNX
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HF_REPO_ID = "hynt/Zipformer-30M-RNNT-6000h"

# Quantised int8 files (default — fast, small, same accuracy)
INT8_FILES = [
    "encoder-epoch-20-avg-10.int8.onnx",
    "decoder-epoch-20-avg-10.int8.onnx",
    "joiner-epoch-20-avg-10.int8.onnx",
]

# Full-precision files (only needed when use_int8=False)
FULL_FILES = [
    "encoder-epoch-20-avg-10.onnx",
    "decoder-epoch-20-avg-10.onnx",
    "joiner-epoch-20-avg-10.onnx",
]

# Always needed
COMMON_FILES = [
    "bpe.model",
    "config.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m asr.download_model",
        description="Download Zipformer-30M-RNNT-6000h from HuggingFace Hub.",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("asr/models/zipformer-30m"),
        metavar="DIR",
        help="Local directory to save the model. (default: asr/models/zipformer-30m)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Also download full-precision ONNX files (encoder ~92 MB). "
             "Not needed if use_int8=True (the default).",
    )
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        metavar="TOKEN",
        help="HuggingFace access token (not required for this public repo).",
    )
    return p.parse_args()


def download(
    model_dir: Path,
    include_full: bool = False,
    token: str | None = None,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is not installed.\n"
            "Run: uv pip install huggingface_hub"
        ) from e

    model_dir.mkdir(parents=True, exist_ok=True)
    files = COMMON_FILES + INT8_FILES + (FULL_FILES if include_full else [])

    print(f"Downloading {HF_REPO_ID} → {model_dir.resolve()}")
    for filename in files:
        dest = model_dir / filename
        if dest.exists():
            print(f"  [skip]  {filename}")
            continue
        print(f"  Downloading {filename} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(model_dir),
            token=token,
        )
        print(f"  [done]  {filename}")

    # Generate tokens.txt from bpe.model (required by sherpa-onnx)
    tokens_path = model_dir / "tokens.txt"
    bpe_path = model_dir / "bpe.model"
    if not tokens_path.exists() and bpe_path.exists():
        print("  Generating tokens.txt from bpe.model ...")
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "sentencepiece is not installed.\n"
                "Run: uv pip install sentencepiece"
            ) from e
        sp = spm.SentencePieceProcessor()
        sp.Load(str(bpe_path))
        with open(tokens_path, "w", encoding="utf-8") as f:
            for i in range(sp.GetPieceSize()):
                f.write(f"{sp.IdToPiece(i)} {i}\n")
        print(f"  [done]  tokens.txt ({sp.GetPieceSize()} tokens)")

    print(f"\nModel ready at: {model_dir.resolve()}")
    print("\nNext steps:")
    print(f"  uv run python -m asr.transcribe your_audio.wav")


def main() -> int:
    args = parse_args()
    try:
        download(args.model_dir, include_full=args.full, token=args.hf_token)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
