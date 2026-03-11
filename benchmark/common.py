from __future__ import annotations

import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def choose_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def resolve_checkpoint_dir(
    checkpoint_dir: Path | None,
    adapter_dir: Path | None,
) -> Path:
    if adapter_dir is not None and checkpoint_dir is not None:
        raise ValueError("Use only one of --adapter_dir or --checkpoint_dir.")
    if adapter_dir is None and checkpoint_dir is None:
        raise ValueError("Provide --checkpoint_dir or --adapter_dir.")
    resolved = checkpoint_dir or adapter_dir
    assert resolved is not None
    return resolved


def is_adapter_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


def is_merged_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "LLM").exists()


def validate_checkpoint_args(
    model_dir: Path,
    checkpoint_dir: Path | None,
    adapter_dir: Path | None,
) -> Path:
    resolved = resolve_checkpoint_dir(checkpoint_dir, adapter_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {resolved}")

    if is_adapter_checkpoint(resolved):
        if not model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")
        return resolved

    if is_merged_checkpoint(resolved):
        return resolved

    raise ValueError(
        f"Unsupported checkpoint layout: {resolved}\n"
        "Expected either:\n"
        "  - adapter checkpoint with adapter_config.json\n"
        "  - merged SparkTTS checkpoint with LLM/ directory"
    )


def validate_synthesis_mode(
    prompt_speech_path: Path | None,
    gender: str | None,
) -> None:
    if gender is None and prompt_speech_path is None:
        raise ValueError(
            "Provide --prompt_speech_path for voice cloning mode, or set --gender "
            "for controllable mode."
        )
    if prompt_speech_path is not None and not prompt_speech_path.exists():
        raise FileNotFoundError(f"prompt_speech_path not found: {prompt_speech_path}")


def load_model_and_tokenizer(
    model_dir: Path,
    checkpoint_dir: Path | None,
    adapter_dir: Path | None,
    device: torch.device,
    dtype: torch.dtype,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = validate_checkpoint_args(model_dir, checkpoint_dir, adapter_dir)

    if is_adapter_checkpoint(resolved):
        tokenizer = AutoTokenizer.from_pretrained(str(resolved))
        base_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir / "LLM"),
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base_model, str(resolved)).to(device)
        print(f"Loaded adapter checkpoint: {resolved}")
        return model, tokenizer, model_dir

    tokenizer = AutoTokenizer.from_pretrained(str(resolved / "LLM"))
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved / "LLM"),
        torch_dtype=dtype,
    ).to(device)
    print(f"Loaded merged checkpoint: {resolved}")
    return model, tokenizer, resolved


def build_tts_pipeline(
    model_dir: Path,
    checkpoint_dir: Path | None,
    adapter_dir: Path | None,
    device: torch.device,
    dtype: torch.dtype,
):
    from cli.SparkTTS import SparkTTS

    model, tokenizer, sparktts_model_dir = load_model_and_tokenizer(
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        adapter_dir=adapter_dir,
        device=device,
        dtype=dtype,
    )
    model.eval()
    tts = SparkTTS(sparktts_model_dir, device)
    tts.model = model
    tts.tokenizer = tokenizer
    return tts


def load_sample_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Sample file must contain a non-empty JSON list.")

    samples: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Sample #{index} must be a JSON object.")
        sample_id = str(item.get("id") or f"sample_{index:02d}")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Sample #{index} must include a non-empty 'text' value.")
        record = dict(item)
        record["id"] = sample_id
        record["text"] = text.strip()
        if "synthesis_text" in record and isinstance(record["synthesis_text"], str):
            record["synthesis_text"] = record["synthesis_text"].strip()
        samples.append(record)
    return samples


def sample_reference_text(sample: dict[str, Any]) -> str:
    synthesis_text = sample.get("synthesis_text")
    if isinstance(synthesis_text, str) and synthesis_text.strip():
        return synthesis_text.strip()
    return str(sample["text"]).strip()


def normalize_for_wer(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("_", " ")
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1] / len(ref_words)


def join_transcription(results: list[Any]) -> str:
    parts = [result.text.strip() for result in results if result.text.strip()]
    return " ".join(parts).strip()


def make_serializable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_run_name(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_numeric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
        }

    ordered = sorted(float(v) for v in values)
    p95_index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * 0.95)) - 1))
    return {
        "count": float(len(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p95": float(ordered[p95_index]),
    }


def synthesize_to_path(
    tts,
    text: str,
    output_path: Path,
    prompt_speech_path: Path | None,
    prompt_text: str | None,
    gender: str | None,
    pitch: str,
    speed: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[np.ndarray, float, float]:
    started_at = time.perf_counter()
    with torch.no_grad():
        wav = tts.inference(
            text=text,
            prompt_speech_path=prompt_speech_path,
            prompt_text=prompt_text,
            gender=gender,
            pitch=pitch if gender else None,
            speed=speed if gender else None,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elapsed = time.perf_counter() - started_at

    wav_np = np.asarray(wav, dtype=np.float32)
    ensure_dir(output_path.parent)
    sf.write(str(output_path), wav_np, tts.sample_rate)
    duration_seconds = float(len(wav_np) / tts.sample_rate) if len(wav_np) else 0.0
    return wav_np, elapsed, duration_seconds
