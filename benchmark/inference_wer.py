#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.common import (
    build_normalizer,
    build_tts_pipeline,
    choose_dtype,
    compute_wer,
    ensure_dir,
    join_transcription,
    load_sample_manifest,
    make_serializable_path,
    normalize_for_wer,
    resolve_device,
    sample_reference_text,
    synthesize_to_path,
    validate_checkpoint_args,
    validate_synthesis_mode,
)

DEFAULT_SAMPLES = PROJECT_ROOT / "benchmark" / "wer_samples.json"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "outputs" / "wer_samples"
DEFAULT_RESULT_CSV = PROJECT_ROOT / "outputs" / "result_wer.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score SparkTTS audio with WER using the hynt/Zipformer-30M-RNNT-6000h "
            "ASR model. The script can either generate audio first or score an "
            "existing directory of wav files."
        )
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models" / "Spark-TTS-0.5B",
        help="SparkTTS base model directory.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=None,
        help=(
            "LoRA adapter checkpoint directory (deprecated alias for "
            "--checkpoint_dir when loading an adapter)."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory to load. Supports either a LoRA adapter "
            "directory (contains adapter_config.json) or a merged SparkTTS "
            "model directory (contains LLM/ and config.yaml)."
        ),
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES,
        help="JSON file containing benchmark samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory where generated wav files will be saved.",
    )
    parser.add_argument(
        "--input_audio_dir",
        type=Path,
        default=None,
        help="Existing wav directory to score. Skips synthesis when provided.",
    )
    parser.add_argument(
        "--result_csv",
        type=Path,
        default=DEFAULT_RESULT_CSV,
        help="CSV path for per-sample WER results.",
    )
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
        help="Transcript of the prompt audio (optional).",
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default=None,
        help="Use controllable mode instead of voice cloning when provided.",
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
        help="Normalize text before synthesis when generating audio.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device string, e.g. "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for more repeatable synthesis.",
    )
    parser.add_argument(
        "--asr_model_dir",
        type=Path,
        default=PROJECT_ROOT / "asr" / "models" / "zipformer-30m",
        help="Directory containing the downloaded Zipformer ONNX files.",
    )
    parser.add_argument(
        "--asr_provider",
        choices=["cpu", "cuda", "coreml"],
        default="cpu",
        help="ONNX execution provider for ASR.",
    )
    parser.add_argument(
        "--asr_threads",
        type=int,
        default=4,
        help="Number of CPU threads used by ASR.",
    )
    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=0.3,
        help="VAD threshold for ASR segmentation.",
    )
    parser.add_argument(
        "--no_int8",
        action="store_true",
        help="Use full-precision ONNX ASR model instead of int8.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.samples.exists():
        raise FileNotFoundError(f"samples file not found: {args.samples}")

    if args.input_audio_dir is not None:
        if not args.input_audio_dir.exists():
            raise FileNotFoundError(f"input_audio_dir not found: {args.input_audio_dir}")
        return

    validate_checkpoint_args(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        adapter_dir=args.adapter_dir,
    )
    validate_synthesis_mode(
        prompt_speech_path=args.prompt_speech_path,
        gender=args.gender,
    )


def build_asr_pipeline(args: argparse.Namespace):
    from asr import ASRConfig, ASRPipeline

    asr_config = ASRConfig(
        model_dir=args.asr_model_dir,
        use_int8=not args.no_int8,
        provider=args.asr_provider,
        num_threads=args.asr_threads,
        vad_threshold=args.vad_threshold,
    )
    return ASRPipeline(asr_config)


def resolve_audio_path(sample: dict[str, Any], audio_dir: Path) -> Path:
    audio_name = sample.get("audio_name")
    if isinstance(audio_name, str) and audio_name.strip():
        return audio_dir / audio_name
    return audio_dir / f"{sample['id']}.wav"


def score_samples_with_asr(
    samples: list[dict[str, Any]],
    audio_dir: Path,
    result_csv: Path,
    pipeline,
) -> tuple[list[dict[str, str]], float]:
    rows: list[dict[str, str]] = []

    for index, sample in enumerate(samples, start=1):
        audio_path = resolve_audio_path(sample, audio_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio for sample {sample['id']}: {audio_path}")

        transcription = join_transcription(pipeline.process_file(audio_path))
        reference_text = sample_reference_text(sample)
        reference_normalized = normalize_for_wer(reference_text)
        hypothesis_normalized = normalize_for_wer(transcription)
        wer = compute_wer(reference_normalized, hypothesis_normalized)

        rows.append(
            {
                "sample_id": str(sample["id"]),
                "reference_text": str(sample.get("text", "")),
                "synthesis_text": reference_text,
                "asr_text": transcription,
                "reference_normalized": reference_normalized,
                "asr_normalized": hypothesis_normalized,
                "wer": f"{wer:.6f}",
                "audio_path": make_serializable_path(audio_path),
            }
        )

        print(f"[{index}/{len(samples)}] {sample['id']} WER={wer:.4f}")
        print(f"  ref: {reference_text}")
        print(f"  hyp: {transcription or '<empty>'}")

    average_wer = (
        sum(float(row["wer"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    rows.append(
        {
            "sample_id": "__average__",
            "reference_text": "",
            "synthesis_text": "",
            "asr_text": "",
            "reference_normalized": "",
            "asr_normalized": "",
            "wer": f"{average_wer:.6f}",
            "audio_path": "",
        }
    )

    ensure_dir(result_csv.parent)
    with result_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample_id",
                "reference_text",
                "synthesis_text",
                "asr_text",
                "reference_normalized",
                "asr_normalized",
                "wer",
                "audio_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows, average_wer


def maybe_generate_audio(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Path]:
    if args.input_audio_dir is not None:
        return samples, args.input_audio_dir

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = choose_dtype(device)
    normalizer = build_normalizer(args.text_normalizer)
    output_dir = ensure_dir(args.output_dir)
    tts = build_tts_pipeline(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        adapter_dir=args.adapter_dir,
        device=device,
        dtype=dtype,
    )

    generated_samples: list[dict[str, Any]] = []
    print(f"Generating audio for {len(samples)} samples...")
    for index, sample in enumerate(samples, start=1):
        synthesis_text = normalizer(sample["text"]) if normalizer else sample["text"]
        audio_name = f"{sample['id']}.wav"
        audio_path = output_dir / audio_name
        synthesize_to_path(
            tts=tts,
            text=synthesis_text,
            output_path=audio_path,
            prompt_speech_path=args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        generated_sample = dict(sample)
        generated_sample["synthesis_text"] = synthesis_text
        generated_sample["audio_name"] = audio_name
        generated_samples.append(generated_sample)
        print(f"[{index}/{len(samples)}] generated {audio_name}")

    return generated_samples, output_dir


def main() -> None:
    args = parse_args()
    validate_args(args)

    samples = load_sample_manifest(args.samples)
    scoring_samples, audio_dir = maybe_generate_audio(args, samples)
    pipeline = build_asr_pipeline(args)
    _, average_wer = score_samples_with_asr(
        samples=scoring_samples,
        audio_dir=audio_dir,
        result_csv=args.result_csv,
        pipeline=pipeline,
    )

    print(f"\nAverage WER: {average_wer:.4f}")
    print(f"Saved CSV: {args.result_csv}")
    print(f"Scored audio directory: {audio_dir}")


if __name__ == "__main__":
    main()
