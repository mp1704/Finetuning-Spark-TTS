"""
Audio file transcription CLI.

Usage:
    uv run python -m asr.transcribe audio.wav
    uv run python -m asr.transcribe audio.wav --show-timestamps
    uv run python -m asr.transcribe audio.wav -o output.txt
    uv run python -m asr.transcribe audio.wav --provider cuda --threads 8
    uv run python -m asr.transcribe audio.wav --model-dir asr/models/zipformer-30m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m asr.transcribe",
        description="Transcribe an audio file using Zipformer RNNT (Vietnamese ASR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "audio",
        type=Path,
        help="Path to the audio file (WAV, FLAC, OGG, etc.)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write transcription to this text file (one utterance per line).",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("asr/models/zipformer-30m"),
        metavar="DIR",
        help="Directory containing the ONNX model files. "
             "(default: asr/models/zipformer-30m)",
    )
    p.add_argument(
        "--no-int8",
        action="store_true",
        default=False,
        help="Use full-precision ONNX instead of int8-quantised (larger, no speed benefit on CPU).",
    )
    p.add_argument(
        "--provider",
        choices=["cpu", "cuda", "coreml"],
        default="cpu",
        help="ONNX execution provider. (default: cpu)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        metavar="N",
        help="Number of CPU threads for ONNX inference. (default: 4)",
    )
    p.add_argument(
        "--show-timestamps",
        action="store_true",
        default=False,
        help="Prefix each line with [start – end] timing info.",
    )
    p.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Silero VAD speech probability threshold 0.0–1.0. (default: 0.5)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy imports so --help is fast even without the packages installed
    from asr.config import ASRConfig
    from asr.pipeline import ASRPipeline

    config = ASRConfig(
        model_dir=args.model_dir,
        use_int8=not args.no_int8,
        provider=args.provider,
        num_threads=args.threads,
        vad_threshold=args.vad_threshold,
    )

    print(f"Loading model from {config.model_dir} ...", file=sys.stderr)
    try:
        pipeline = ASRPipeline(config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Transcribing {args.audio} ...", file=sys.stderr)
    try:
        results = pipeline.process_file(args.audio)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not results:
        print("Warning: no speech detected in the audio file.", file=sys.stderr)
        return 0

    lines: list[str] = []
    for r in results:
        if args.show_timestamps:
            line = (
                f"[{r.segment.start_seconds:.2f}s – {r.segment.end_seconds:.2f}s]  "
                f"{r.text}"
            )
        else:
            line = r.text
        lines.append(line)
        print(line)

    if args.output:
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
