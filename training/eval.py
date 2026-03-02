"""
Evaluate a fine-tuned SparkTTS checkpoint using UTMOSv2 (MOS) and WER (Whisper).

For each sample in the test set:
  1. Generate speech from text + reference speaker audio
  2. Score with UTMOSv2 → MOS float (1–5)
  3. Transcribe with Whisper large-v3 → compute WER vs ground truth

Outputs: eval_results.csv + aggregate means printed to stdout.

Usage:
    python training/eval.py \
        --config training/configs/train_phase1.yaml \
        --checkpoint checkpoints/phase1/checkpoint-5000 \
        --output eval_results.csv \
        [--num_samples 100]
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sparktts(model_dir: str, checkpoint: str, device: torch.device):
    """Load base SparkTTS with LoRA weights merged from checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sparktts.models.audio_tokenizer import BiCodecTokenizer

    llm_dir = f"{model_dir}/LLM"
    tokenizer = AutoTokenizer.from_pretrained(llm_dir)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        llm_dir, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model = model.merge_and_unload()
    model.to(device).eval()

    audio_tokenizer = BiCodecTokenizer(model_dir=model_dir, device=device)

    return model, tokenizer, audio_tokenizer


def generate_wav(text: str, ref_audio_path: str, model, tokenizer,
                 audio_tokenizer, device, generation_kwargs: dict) -> np.ndarray:
    """Generate speech for one sample (mirrors cli/SparkTTS.py:inference)."""
    import re

    from sparktts.utils.token_parser import TASK_TOKEN_MAP

    global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(ref_audio_path)
    global_str = "".join(
        f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().tolist()
    )

    prompt = (
        f"{TASK_TOKEN_MAP['tts']}"
        "<|start_content|>"
        f"{text}"
        "<|end_content|>"
        "<|start_global_token|>"
        f"{global_str}"
        "<|end_global_token|>"
        "<|start_semantic_token|>"
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=generation_kwargs.get("max_new_tokens", 3000),
            do_sample=True,
            top_k=generation_kwargs.get("top_k", 50),
            top_p=generation_kwargs.get("top_p", 0.95),
            temperature=generation_kwargs.get("temperature", 0.8),
        )

    gen_ids = gen_ids[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    pred_semantic = (
        torch.tensor(
            [int(t) for t in __import__("re").findall(r"bicodec_semantic_(\d+)", decoded)]
        ).long().unsqueeze(0)
    )

    wav = audio_tokenizer.detokenize(
        global_token_ids.to(device).squeeze(0),
        pred_semantic.to(device),
    )
    return wav


def load_utmos():
    try:
        import utmosv2
        model = utmosv2.create_model(pretrained=True)
        model.eval()
        return model
    except ImportError:
        raise ImportError(
            "UTMOSv2 not installed. Run:\n"
            "  uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git"
        )


def load_whisper(model_name: str = "large-v3"):
    try:
        import whisper
        return whisper.load_model(model_name)
    except ImportError:
        raise ImportError("openai-whisper not installed. Run: uv pip install openai-whisper")


def compute_wer(reference: str, hypothesis: str) -> float:
    from jiwer import wer
    return wer(reference.lower().strip(), hypothesis.lower().strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint directory")
    parser.add_argument("--output", default="eval_results.csv")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on first N samples (default: all)")
    parser.add_argument("--whisper_model", default=None,
                        help="Override whisper model name (default from config)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = cfg["model_dir"]
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg["data"]

    whisper_model_name = args.whisper_model or eval_cfg.get("whisper_model", "large-v3")
    whisper_lang = eval_cfg.get("whisper_language", "vi")
    device = torch.device(args.device)

    # Load models
    print("Loading SparkTTS checkpoint ...")
    model, tokenizer, audio_tokenizer = load_sparktts(model_dir, args.checkpoint, device)

    print("Loading UTMOSv2 ...")
    utmos_model = load_utmos()

    print(f"Loading Whisper {whisper_model_name} ...")
    asr_model = load_whisper(whisper_model_name)

    # Load eval metadata
    metadata_path = data_cfg["metadata"]
    tok_path = metadata_path.replace(".jsonl", "_tokenized.jsonl")
    if Path(tok_path).exists():
        metadata_path = tok_path

    with open(metadata_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Hold-out test set (last test_split fraction, same split as training)
    test_frac = eval_cfg.get("test_split", 0.05)
    n_test = max(1, int(len(records) * test_frac))
    test_records = records[-n_test:]

    if args.num_samples:
        test_records = test_records[: args.num_samples]

    print(f"Evaluating on {len(test_records)} samples ...")

    generation_kwargs = {
        "max_new_tokens": eval_cfg.get("max_new_tokens", 3000),
        "top_k": eval_cfg.get("top_k", 50),
        "top_p": eval_cfg.get("top_p", 0.95),
        "temperature": eval_cfg.get("temperature", 0.8),
    }

    results = []
    mos_scores, wer_scores = [], []

    with tempfile.TemporaryDirectory() as tmpdir:
        for rec in tqdm(test_records, desc="Evaluating"):
            text = rec["text"].strip()
            ref_audio = rec["audio_path"]

            try:
                wav = generate_wav(
                    text, ref_audio, model, tokenizer, audio_tokenizer,
                    device, generation_kwargs,
                )
            except Exception as e:
                print(f"Generation error: {e}")
                continue

            wav_path = os.path.join(tmpdir, "gen.wav")
            sf.write(wav_path, wav, 16000)

            # UTMOSv2 MOS
            try:
                mos = float(utmos_model.predict(input_path=wav_path))
            except Exception as e:
                print(f"UTMOSv2 error: {e}")
                mos = float("nan")

            # WER via Whisper
            try:
                asr_result = asr_model.transcribe(wav_path, language=whisper_lang)
                hypothesis = asr_result["text"]
                w = compute_wer(text, hypothesis)
            except Exception as e:
                print(f"Whisper/WER error: {e}")
                hypothesis, w = "", float("nan")

            results.append({
                "audio_path": ref_audio,
                "reference": text,
                "hypothesis": hypothesis,
                "utmos_mos": mos,
                "wer": w,
            })

            if not np.isnan(mos):
                mos_scores.append(mos)
            if not np.isnan(w):
                wer_scores.append(w)

    # Write CSV
    output_path = args.output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*50}")
    print(f"Results saved to: {output_path}")
    print(f"Samples evaluated: {len(results)}")
    if mos_scores:
        print(f"Mean UTMOSv2 MOS : {sum(mos_scores)/len(mos_scores):.3f}")
    if wer_scores:
        print(f"Mean WER          : {sum(wer_scores)/len(wer_scores):.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
