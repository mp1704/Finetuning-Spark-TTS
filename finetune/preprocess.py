"""
Offline preprocessing: tokenize audio with BiCodec and save semantic + global
token indices to JSONL for fast training data loading.

Supports two dataset sources:

  LJSpeech (local):
      uv run python -m finetune.preprocess \\
          --source ljspeech \\
          --model_dir pretrained_models/Spark-TTS-0.5B \\
          --data_dir LJSpeech-1.1 \\
          --device cuda

  HuggingFace dataset with Vinorm text normalisation (e.g. pnnbao-ump/ngochuyen_voice):
      uv run python -m finetune.preprocess \\
          --source huggingface \\
          --hf_dataset pnnbao-ump/ngochuyen_voice \\
          --hf_audio_col audio \\
          --hf_text_col transcription \\
          --text_normalizer vinorm \\
          --output_dir data/ngochuyen \\
          --val_size 500 \\
          --model_dir pretrained_models/Spark-TTS-0.5B \\
          --device cuda

Text normaliser options:
  none    — no normalisation (default)
  vinorm  — Vietnamese NSW normalisation via Vinorm (pip install vinorm)
            Converts numbers, dates, abbreviations → spoken syllables.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soxr
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize


# ── text normaliser ───────────────────────────────────────────────────────────

def build_normalizer(name: str):
    """Return a text normalisation callable, or None.

    Args:
        name: "none" | "vinorm"

    Returns:
        Callable[[str], str] or None
    """
    if name == "none":
        return None

    if name == "vinorm":
        try:
            from vinorm import TTSnorm
        except ImportError:
            print("[ERROR] vinorm not installed. Run: uv pip install vinorm")
            sys.exit(1)
        # TTSnorm(text) returns the normalised string directly.
        return lambda text: TTSnorm(text)

    raise ValueError(f"Unknown text_normalizer: {name!r}")


# ── helpers ───────────────────────────────────────────────────────────────────

def tokenize_array(
    wav: np.ndarray,
    src_sr: int,
    tokenizer: BiCodecTokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a raw numpy audio array without writing to disk.

    Args:
        wav:       1-D float32 numpy array (mono).
        src_sr:    Sample rate of `wav`.
        tokenizer: Initialised BiCodecTokenizer.

    Returns:
        (global_tokens, semantic_tokens) as plain Python int lists.
    """
    target_sr: int = tokenizer.config["sample_rate"]  # 16000

    # 1. Resample to target SR if necessary
    if src_sr != target_sr:
        wav = soxr.resample(wav, src_sr, target_sr, quality="VHQ")

    # 2. Volume normalise (mirrors load_audio behaviour)
    if tokenizer.config.get("volume_normalize", True):
        wav = audio_volume_normalize(wav)

    # 3. Build ref clip (speaker embedding input)
    wav_ref = tokenizer.get_ref_clip(wav)
    wav_ref_t = torch.from_numpy(wav_ref).unsqueeze(0).float().to(tokenizer.device)

    # 4. Extract wav2vec2 features
    feat = tokenizer.extract_wav2vec2_features(wav)  # uses processor internally

    batch = {
        "wav": torch.from_numpy(wav).unsqueeze(0).float().to(tokenizer.device),
        "ref_wav": wav_ref_t,
        "feat": feat.to(tokenizer.device),
    }

    with torch.no_grad():
        semantic_tokens, global_tokens = tokenizer.model.tokenize(batch)

    return (
        global_tokens.squeeze().cpu().tolist(),
        semantic_tokens.squeeze().cpu().tolist(),
    )


# ── LJSpeech ─────────────────────────────────────────────────────────────────

def read_ljspeech_csv(csv_path: Path) -> List[dict]:
    """Read LJSpeech pipe-delimited CSV (no header row)."""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            sample_id = parts[0]
            text = parts[2] if len(parts) >= 3 else parts[1]
            samples.append({"id": sample_id, "text": text})
    return samples


def process_ljspeech_split(
    split: str,
    data_dir: Path,
    tokenizer: BiCodecTokenizer,
    output_dir: Path,
    normalizer=None,
):
    csv_path = data_dir / f"metadata_{split}.csv"
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping split '{split}'")
        return

    samples = read_ljspeech_csv(csv_path)
    print(f"LJSpeech — processing {len(samples)} samples for split '{split}'")

    out_path = output_dir / f"{split}_tokens.jsonl"
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc=split):
            wav_path = data_dir / "wavs" / f"{sample['id']}.wav"
            if not wav_path.exists():
                skipped += 1
                continue
            try:
                with torch.no_grad():
                    global_tokens, semantic_tokens = tokenizer.tokenize(str(wav_path))
                text = normalizer(sample["text"]) if normalizer else sample["text"]
                record = {
                    "id": sample["id"],
                    "text": text,
                    "global_tokens": global_tokens.squeeze().cpu().tolist(),
                    "semantic_tokens": semantic_tokens.squeeze().cpu().tolist(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\n[ERROR] {sample['id']}: {e}")
                skipped += 1

    print(f"Saved {len(samples) - skipped} records → {out_path}  (skipped {skipped})")


# ── HuggingFace dataset ───────────────────────────────────────────────────────

def process_huggingface(
    hf_dataset: str,
    audio_col: str,
    text_col: str,
    val_size: int,
    tokenizer: BiCodecTokenizer,
    output_dir: Path,
    hf_split: str = "train",
    normalizer=None,
):
    try:
        import io
        import soundfile as sf
        from datasets import Audio, load_dataset
    except ImportError as exc:
        print(f"[ERROR] Missing package: {exc}. Run: uv pip install datasets soundfile")
        sys.exit(1)

    print(f"Loading HuggingFace dataset '{hf_dataset}' (split='{hf_split}') ...")
    # trust_remote_code removed in datasets ≥ 2.x; load without it.
    ds = load_dataset(hf_dataset, split=hf_split)

    # Disable the automatic torchcodec/torchaudio decode pipeline.
    # decode=False returns {"bytes": bytes | None, "path": str | None} instead
    # of a decoded array, letting us use soundfile directly.
    ds = ds.cast_column(audio_col, Audio(decode=False))
    print(f"  Total samples: {len(ds)}")

    # Deterministic train/val split
    val_n = min(val_size, len(ds) // 10)
    train_n = len(ds) - val_n
    splits = {
        "train": ds.select(range(train_n)),
        "val": ds.select(range(train_n, len(ds))),
    }
    print(f"  Split → train: {train_n}, val: {val_n}")

    for split_name, subset in splits.items():
        out_path = output_dir / f"{split_name}_tokens.jsonl"
        skipped = 0

        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, sample in enumerate(tqdm(subset, desc=split_name)):
                try:
                    # decode=False gives {"bytes": bytes|None, "path": str|None}
                    audio_obj = sample[audio_col]
                    raw_bytes = audio_obj.get("bytes")
                    raw_path  = audio_obj.get("path")

                    if raw_bytes is not None:
                        wav, src_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                    elif raw_path is not None:
                        wav, src_sr = sf.read(raw_path, dtype="float32")
                    else:
                        raise ValueError("Audio entry has neither 'bytes' nor 'path'.")

                    wav = np.array(wav, dtype=np.float32)
                    # Mono-ify if stereo
                    if wav.ndim == 2:
                        wav = wav.mean(axis=1)

                    raw_text: str = sample[text_col].strip()
                    text: str = normalizer(raw_text) if normalizer else raw_text
                    sample_id: str = sample.get("file_name", str(i)).strip()

                    global_toks, semantic_toks = tokenize_array(wav, src_sr, tokenizer)

                    record = {
                        "id": sample_id,
                        "text": text,
                        "global_tokens": global_toks,
                        "semantic_tokens": semantic_toks,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"\n[ERROR] sample {i}: {e}")
                    skipped += 1

        print(f"Saved {len(subset) - skipped} records → {out_path}  (skipped {skipped})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize audio dataset with BiCodec")
    parser.add_argument(
        "--source",
        choices=["ljspeech", "huggingface"],
        default="ljspeech",
        help="Dataset source",
    )
    parser.add_argument(
        "--model_dir",
        default="pretrained_models/Spark-TTS-0.5B",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # LJSpeech options
    parser.add_argument("--data_dir", default="LJSpeech-1.1")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="LJSpeech splits to process",
    )

    # HuggingFace options
    parser.add_argument("--hf_dataset", default="pnnbao-ump/ngochuyen_voice")
    parser.add_argument("--hf_split", default="train", help="HF dataset split to load")
    parser.add_argument("--hf_audio_col", default="audio")
    parser.add_argument("--hf_text_col", default="transcription")
    parser.add_argument(
        "--val_size", type=int, default=500,
        help="Number of samples to hold out for validation (HuggingFace source only)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write JSONL files. Defaults to --data_dir for LJSpeech, "
             "or data/<hf_dataset_name> for HuggingFace.",
    )

    # Text normalisation
    parser.add_argument(
        "--text_normalizer",
        choices=["none", "vinorm"],
        default="none",
        help=(
            "Text normaliser to apply before saving transcripts. "
            "'vinorm' converts Vietnamese numbers/dates/abbreviations to spoken form "
            "(requires: uv pip install vinorm)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    normalizer = build_normalizer(args.text_normalizer)
    if normalizer:
        print(f"Text normaliser: {args.text_normalizer}")

    print(f"Loading BiCodecTokenizer from '{args.model_dir}' on {args.device} ...")
    tokenizer = BiCodecTokenizer(model_dir=args.model_dir, device=device)

    if args.source == "ljspeech":
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir) if args.output_dir else data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            process_ljspeech_split(split, data_dir, tokenizer, output_dir, normalizer=normalizer)

    elif args.source == "huggingface":
        dataset_name = args.hf_dataset
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # e.g. "pnnbao-ump/ngochuyen_voice" → "data/ngochuyen_voice"
            output_dir = Path("data") / dataset_name.split("/")[-1]
        output_dir.mkdir(parents=True, exist_ok=True)
        process_huggingface(
            hf_dataset=dataset_name,
            audio_col=args.hf_audio_col,
            text_col=args.hf_text_col,
            val_size=args.val_size,
            tokenizer=tokenizer,
            output_dir=output_dir,
            hf_split=args.hf_split,
            normalizer=normalizer,
        )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
