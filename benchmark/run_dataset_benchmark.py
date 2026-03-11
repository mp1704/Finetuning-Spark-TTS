#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.common import (
    build_normalizer,
    build_tts_pipeline,
    choose_dtype,
    default_run_name,
    ensure_dir,
    make_serializable_path,
    resolve_device,
    summarize_numeric,
    synthesize_to_path,
    validate_checkpoint_args,
    validate_synthesis_mode,
    write_json,
)
from benchmark.hf_dataset import load_hf_validation_samples, write_hf_manifest
from benchmark.inference_wer import build_asr_pipeline, score_samples_with_asr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a dataset-driven SparkTTS benchmark on a Hugging Face validation "
            "split and collect WER, UTMOSv2, and RTF."
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
        help="Deprecated alias for --checkpoint_dir when loading an adapter.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Checkpoint directory to benchmark.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="linhtran92/viet_bud500",
        help="Hugging Face dataset ID.",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="validation",
        help="Dataset split to benchmark.",
    )
    parser.add_argument(
        "--hf_text_col",
        type=str,
        default="transcription",
        help="Dataset text column.",
    )
    parser.add_argument(
        "--hf_id_col",
        type=str,
        default="",
        help="Optional dataset ID column. Leave empty to derive IDs from row index.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional max number of validation samples to benchmark.",
    )
    parser.add_argument(
        "--prompt_speech_path",
        type=Path,
        default=None,
        help="Prompt wav path for shared-prompt voice cloning mode.",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Transcript of the shared prompt audio (optional).",
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default=None,
        help="Use controllable mode instead of prompt-based voice cloning.",
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
        help="Normalize dataset text before synthesis.",
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
        help="Random seed for repeatable synthesis.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmarks",
        help="Directory where benchmark runs are created.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run directory name. Defaults to a timestamped name.",
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
    parser.add_argument(
        "--utmos_config",
        type=str,
        default="fusion_stage3",
        help="UTMOSv2 config name passed to benchmark/inference_utmos.py.",
    )
    parser.add_argument(
        "--utmos_num_workers",
        type=int,
        default=4,
        help="Number of workers for UTMOSv2 dataloader.",
    )
    parser.add_argument(
        "--utmos_predict_dataset",
        type=str,
        default="sarulab",
        help="UTMOSv2 predict dataset argument.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_checkpoint_args(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        adapter_dir=args.adapter_dir,
    )
    validate_synthesis_mode(
        prompt_speech_path=args.prompt_speech_path,
        gender=args.gender,
    )
    ensure_dir(args.output_root)


def write_rtf_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "reference_text",
                "synthesis_text",
                "audio_path",
                "generation_seconds",
                "audio_seconds",
                "rtf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_utmos(
    audio_dir: Path,
    out_csv: Path,
    config_name: str,
    num_workers: int,
    predict_dataset: str,
) -> list[dict[str, str]]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark" / "inference_utmos.py"),
        "--config",
        config_name,
        "--input_dir",
        str(audio_dir),
        "--out_path",
        str(out_csv),
        "--num_workers",
        str(num_workers),
        "--predict_dataset",
        predict_dataset,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    with out_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def main() -> None:
    args = parse_args()
    validate_args(args)
    resolved_checkpoint_dir = validate_checkpoint_args(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        adapter_dir=args.adapter_dir,
    )

    run_name = args.run_name or default_run_name("viet-bud500-validation")
    run_dir = ensure_dir(args.output_root / run_name)
    generated_dir = ensure_dir(run_dir / "generated_wavs")
    manifest_path = run_dir / "samples.json"
    rtf_csv = run_dir / "rtf.csv"
    wer_csv = run_dir / "wer.csv"
    utmos_csv = run_dir / "utmos.csv"
    summary_json = run_dir / "summary.json"

    samples = load_hf_validation_samples(
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        text_col=args.hf_text_col,
        id_col=args.hf_id_col or None,
        sample_limit=args.sample_limit,
    )
    print(f"Loaded {len(samples)} samples from {args.hf_dataset}:{args.hf_split}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = choose_dtype(device)
    normalizer = build_normalizer(args.text_normalizer)
    tts = build_tts_pipeline(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        adapter_dir=args.adapter_dir,
        device=device,
        dtype=dtype,
    )

    manifest_rows: list[dict[str, object]] = []
    rtf_rows: list[dict[str, str]] = []
    rtf_values: list[float] = []
    generation_seconds: list[float] = []
    audio_seconds: list[float] = []

    for index, sample in enumerate(samples, start=1):
        synthesis_text = normalizer(sample["text"]) if normalizer else sample["text"]
        audio_name = f"{sample['id']}.wav"
        audio_path = generated_dir / audio_name
        _, elapsed, duration_seconds = synthesize_to_path(
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
        rtf = (elapsed / duration_seconds) if duration_seconds > 0 else 0.0

        manifest_row = dict(sample)
        manifest_row["synthesis_text"] = synthesis_text
        manifest_row["audio_name"] = audio_name
        manifest_row["audio_path"] = make_serializable_path(audio_path)
        manifest_rows.append(manifest_row)

        rtf_rows.append(
            {
                "sample_id": str(sample["id"]),
                "reference_text": str(sample["text"]),
                "synthesis_text": synthesis_text,
                "audio_path": make_serializable_path(audio_path),
                "generation_seconds": f"{elapsed:.6f}",
                "audio_seconds": f"{duration_seconds:.6f}",
                "rtf": f"{rtf:.6f}",
            }
        )
        rtf_values.append(rtf)
        generation_seconds.append(elapsed)
        audio_seconds.append(duration_seconds)
        print(
            f"[{index}/{len(samples)}] generated {audio_name} "
            f"(audio={duration_seconds:.2f}s, wall={elapsed:.2f}s, rtf={rtf:.4f})"
        )

    write_hf_manifest(manifest_path, manifest_rows)
    write_rtf_csv(rtf_csv, rtf_rows)

    asr_pipeline = build_asr_pipeline(args)
    _, average_wer = score_samples_with_asr(
        samples=manifest_rows,
        audio_dir=generated_dir,
        result_csv=wer_csv,
        pipeline=asr_pipeline,
    )

    utmos_rows = run_utmos(
        audio_dir=generated_dir,
        out_csv=utmos_csv,
        config_name=args.utmos_config,
        num_workers=args.utmos_num_workers,
        predict_dataset=args.utmos_predict_dataset,
    )
    utmos_values = [
        float(row["mos"])
        for row in utmos_rows
        if row.get("mos") is not None and str(row["mos"]).strip()
    ]

    summary = {
        "run_name": run_name,
        "dataset": {
            "name": args.hf_dataset,
            "split": args.hf_split,
            "text_col": args.hf_text_col,
            "id_col": args.hf_id_col or None,
            "sample_count": len(manifest_rows),
            "sample_limit": args.sample_limit,
        },
        "checkpoint": {
            "checkpoint_dir": make_serializable_path(resolved_checkpoint_dir),
            "adapter_dir": (
                make_serializable_path(args.adapter_dir)
                if args.adapter_dir is not None
                else None
            ),
            "model_dir": make_serializable_path(args.model_dir),
        },
        "inference": {
            "mode": "voice_cloning" if args.prompt_speech_path is not None else "controllable",
            "prompt_speech_path": (
                make_serializable_path(args.prompt_speech_path)
                if args.prompt_speech_path is not None
                else None
            ),
            "prompt_text": args.prompt_text,
            "gender": args.gender,
            "pitch": args.pitch,
            "speed": args.speed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "text_normalizer": args.text_normalizer,
            "device": str(device),
            "seed": args.seed,
        },
        "artifacts": {
            "manifest": make_serializable_path(manifest_path),
            "generated_wavs": make_serializable_path(generated_dir),
            "rtf_csv": make_serializable_path(rtf_csv),
            "wer_csv": make_serializable_path(wer_csv),
            "utmos_csv": make_serializable_path(utmos_csv),
        },
        "metrics": {
            "wer": {
                "average": average_wer,
            },
            "utmos": {
                "mean_mos": (sum(utmos_values) / len(utmos_values)) if utmos_values else 0.0,
                "count": len(utmos_values),
            },
            "rtf": {
                "rtf": summarize_numeric(rtf_values),
                "generation_seconds": summarize_numeric(generation_seconds),
                "audio_seconds": summarize_numeric(audio_seconds),
            },
        },
    }
    write_json(summary_json, summary)

    print(f"\nBenchmark complete: {run_dir}")
    print(f"  WER average: {average_wer:.4f}")
    print(
        "  UTMOS mean: "
        f"{summary['metrics']['utmos']['mean_mos']:.4f}"
    )
    print(
        "  RTF mean: "
        f"{summary['metrics']['rtf']['rtf']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
