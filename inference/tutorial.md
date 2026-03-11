# Inference Tutorial

This tutorial shows how to run inference for the finetuned SparkTTS model in this repository.

## Prerequisites

- `uv` installed
- dependencies installed from `requirements.txt`
- pretrained SparkTTS model available at `pretrained_models/Spark-TTS-0.5B/`
- one of the following checkpoints:
  - a LoRA adapter directory such as `checkpoints/vieneu-lora`
  - a merged checkpoint directory such as `checkpoints/vieneu-merged`

Install dependencies:

```bash
uv pip install -r requirements.txt
```

## Entry point

Use:

```bash
uv run python inference/run_finetuned_inference.py --help
```

The script supports both:

- `--checkpoint_dir` for the new generic checkpoint path
- `--adapter_dir` as a backward-compatible alias for adapter checkpoints

## 1. Run with a LoRA adapter checkpoint

Voice cloning mode with a reference audio:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --checkpoint_dir checkpoints/vieneu-lora \
  --text "Nhà trường có năm giảng viên tham gia đoàn cứu hộ quốc tế sau thảm họa động đất tại Thổ Nhĩ Kỳ." \
  --prompt_speech_path ref2.wav \
  --output_path outputs/adapter_infer.wav
```

Controllable mode without a reference audio:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --checkpoint_dir checkpoints/vieneu-lora \
  --text "Xin chào, đây là một câu kiểm thử cho mô hình SparkTTS." \
  --gender female \
  --pitch moderate \
  --speed moderate \
  --output_path outputs/adapter_control.wav
```

## 2. Run with a merged checkpoint

If you already merged the adapter into a standalone model, you can run inference directly from that checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
  --checkpoint_dir checkpoints/vieneu-merged \
  --text "Xin chào, đây là một câu kiểm thử cho merged checkpoint." \
  --prompt_speech_path ref2.wav \
  --output_path outputs/merged_infer.wav
```

In merged mode, `--model_dir` is usually not needed because the script uses the merged checkpoint directory as the SparkTTS model root.

## 3. Use text normalization

If your input contains written forms such as numbers, dates, or abbreviations, you can enable:

```bash
--text_normalizer vinorm
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --checkpoint_dir checkpoints/vieneu-lora \
  --text "Ngày 25/12/2024, nhiệt độ đạt 32 độ C." \
  --gender female \
  --text_normalizer vinorm \
  --output_path outputs/normalized_infer.wav
```

If your text is already written in spoken form, prefer:

```bash
--text_normalizer none
```

## 4. Important arguments

| Flag | Meaning |
|------|---------|
| `--checkpoint_dir` | Generic checkpoint path. Supports both adapter and merged layouts |
| `--adapter_dir` | Old adapter-only alias |
| `--model_dir` | Base SparkTTS model directory, needed for adapter mode |
| `--text` | Text to synthesize |
| `--prompt_speech_path` | Reference audio for voice cloning |
| `--prompt_text` | Transcript of reference audio, optional |
| `--gender` | Enables controllable mode |
| `--pitch` | Speaking pitch for controllable mode |
| `--speed` | Speaking speed for controllable mode |
| `--temperature` | Sampling temperature |
| `--top_k` | Top-k sampling |
| `--top_p` | Top-p sampling |
| `--device` | `cuda`, `cuda:0`, or `cpu` |
| `--output_path` | Output `.wav` path |

## 5. Rules for mode selection

Choose one of these:

- voice cloning: provide `--prompt_speech_path`
- controllable mode: provide `--gender`

If neither is provided, the script will stop with an error.

## 6. Example shell wrapper

There is already a helper script:

```bash
bash script/03_inference.sh
```

That file contains examples for adapter inference and can be edited for your own checkpoint paths and output names.

## 7. Troubleshooting

### `FileNotFoundError: checkpoint_dir not found`

Check the path you passed to `--checkpoint_dir` or `--adapter_dir`.

### CUDA requested but script falls back to CPU

Your current environment does not expose CUDA to PyTorch. Test with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### `vinorm` import fails

Install it with:

```bash
uv pip install vinorm
```

If your environment is on Python 3.12 and `vinorm` still fails, use `--text_normalizer none` until that package is compatible.

### Adapter checkpoint loads but merged checkpoint does not

Make sure the merged directory contains `LLM/` and the copied SparkTTS layout produced by `finetune/merge_lora.py`.

## 8. Merge a checkpoint first

If you need to create a merged checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python finetune/merge_lora.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --adapter_dir checkpoints/vieneu-lora \
  --output_dir checkpoints/vieneu-merged
```

Then run inference with:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
  --checkpoint_dir checkpoints/vieneu-merged \
  --text "Xin chào" \
  --prompt_speech_path ref2.wav \
  --output_path outputs/merged_infer.wav
```
