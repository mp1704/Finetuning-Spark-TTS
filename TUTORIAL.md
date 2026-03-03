# SparkTTS Finetuning — Tutorial

## Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (≥ 16 GB VRAM recommended for Qwen2.5-0.5B + LoRA)
- [uv](https://github.com/astral-sh/uv) installed
- Pretrained SparkTTS model at `pretrained_models/Spark-TTS-0.5B/`
- LJSpeech dataset at `LJSpeech-1.1/` (run `bash download_ljspeech.sh` if missing), **or** a HuggingFace dataset (e.g. `pnnbao-ump/ngochuyen_voice`)

---

## Step 1 — Install dependencies

```bash
uv pip install peft wandb python-dotenv
```

For HuggingFace datasets (e.g. `pnnbao-ump/ngochuyen_voice`):

```bash
uv pip install datasets
```

For Vietnamese text normalisation with Vinorm:

```bash
uv pip install vinorm
```

All other dependencies (`torch`, `transformers`, `omegaconf`, etc.) are already in `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

---

## Step 2 — Configure WandB

Create a `.env` file in the project root with your WandB API key:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

The training script loads this automatically via `python-dotenv`. You can also set the variable in your shell if you prefer:

```bash
export WANDB_API_KEY=your_key_here
```

---

## Step 3 — Preprocess dataset (one-time)

This step runs each audio file through the frozen BiCodec to extract discrete token indices and writes them to JSONL files. It only needs to be run once.

### Option A — LJSpeech (local files)

```bash
uv run python -m finetune.preprocess \
    --source ljspeech \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --data_dir LJSpeech-1.1 \
    --device cuda
```

On a single A100 this takes roughly 30–60 minutes for all 13 100 samples.

**Outputs:**
```
LJSpeech-1.1/train_tokens.jsonl   # 12 000 records
LJSpeech-1.1/val_tokens.jsonl     #  1 100 records
```

**Sanity check:**
```bash
wc -l LJSpeech-1.1/train_tokens.jsonl   # should print 12000
wc -l LJSpeech-1.1/val_tokens.jsonl     # should print 1100
```

### Option B — HuggingFace dataset (pipeline validation)

Example with `pnnbao-ump/ngochuyen_voice` (Vietnamese, 7 540 samples, 24 kHz):

```bash
uv run python -m finetune.preprocess \
    --source huggingface \
    --hf_dataset pnnbao-ump/ngochuyen_voice \
    --hf_audio_col audio \
    --hf_text_col transcription \
    --text_normalizer vinorm \
    --val_size 500 \
    --output_dir data/ngochuyen \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --device cuda
```

Audio is resampled from 24 kHz → 16 kHz in memory (no temp files). The last `--val_size` samples are held out for validation. `--text_normalizer vinorm` applies [Vinorm](https://github.com/v-nhandt21/Vinorm) to convert numbers, dates, and abbreviations into pronounceable Vietnamese syllables before saving.

**Outputs:**
```
data/ngochuyen/train_tokens.jsonl   # 7 040 records
data/ngochuyen/val_tokens.jsonl     #   500 records
```

Each record (both sources) looks like:
```json
{"id": "LJ001-0001", "text": "Printing, in the only sense...", "global_tokens": [12, 45, ...], "semantic_tokens": [302, 18, ...]}
```

---

## Step 4 — Configure training

For the HuggingFace dataset, update the JSONL paths in [finetune/config.yaml](finetune/config.yaml):

```yaml
data:
  train_jsonl: data/ngochuyen/train_tokens.jsonl
  val_jsonl:   data/ngochuyen/val_tokens.jsonl
```

Other keys to adjust:

| Key | Default | Notes |
|-----|---------|-------|
| `training.bf16` | `true` | Set `false` + `fp16: true` on Ampere-older GPUs |
| `training.per_device_train_batch_size` | `4` | Reduce if OOM |
| `training.gradient_accumulation_steps` | `8` | Effective batch = 4 × 8 = 32 |
| `training.num_train_epochs` | `3` | ~3 epochs is usually enough |
| `lora.r` | `16` | Higher = more capacity, more memory |
| `wandb.project` | `sparktts-finetune` | Your WandB project name |
| `wandb.run_name` | `ljspeech-lora` | WandB run name |

---

## Step 5 — Train

```bash
uv run python -m finetune.train --config finetune/config.yaml
```

Progress is logged to your WandB project. Checkpoints are saved to `checkpoints/ljspeech_lora/`.

**Quick smoke-test (first ~50 steps only):**

```bash
# Temporarily override epochs in config, or pass via env override
uv run python -m finetune.train --config finetune/config.yaml
# (edit config.yaml: num_train_epochs: 0.05, then revert after test)
```

---

## Step 6 — Run inference with the finetuned model

Load the base LLM and merge the LoRA adapter before inference:

```python
from pathlib import Path
import torch
import soundfile as sf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparktts.models.audio_tokenizer import BiCodecTokenizer

MODEL_DIR = "pretrained_models/Spark-TTS-0.5B"
ADAPTER_DIR = "checkpoints/ljspeech_lora"
device = torch.device("cuda")

# Load base model + adapter
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR + "/LLM", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device)
model.eval()

# Use the existing SparkTTS inference pipeline
# (replace model / tokenizer inside a SparkTTS instance, or call directly)
from cli.SparkTTS import SparkTTS
tts = SparkTTS(MODEL_DIR, device)
tts.model = model
tts.tokenizer = tokenizer

wav = tts.inference(
    text="Hello, this is a test of the finetuned SparkTTS model.",
    prompt_speech_path="LJSpeech-1.1/wavs/LJ050-0001.wav",  # any LJSpeech clip
)
sf.write("output_finetuned.wav", wav, 16000)
print("Saved output_finetuned.wav")
```

Or run the dedicated script in `inferece/`:

```bash
uv run python inferece/run_finetuned_inference.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --adapter_dir checkpoints/ngochuyen-demo \
  --text "Xin chao, day la ban thu nghiem voi checkpoint moi." \
  --prompt_speech_path LJSpeech-1.1/wavs/LJ050-0001.wav \
  --output_path outputs/output_finetuned.wav
```

---

## Directory structure after setup

```
spa1/
├── finetune/
│   ├── __init__.py
│   ├── preprocess.py        # Step 3 — supports both LJSpeech and HuggingFace
│   ├── dataset.py           # Dataset + collate (format-agnostic JSONL)
│   ├── train.py             # Step 5
│   └── config.yaml          # Step 4
├── LJSpeech-1.1/
│   ├── wavs/
│   ├── metadata_train.csv
│   ├── metadata_val.csv
│   ├── train_tokens.jsonl   # generated by preprocess (LJSpeech)
│   └── val_tokens.jsonl
├── data/
│   └── ngochuyen/
│       ├── train_tokens.jsonl  # generated by preprocess (HuggingFace)
│       └── val_tokens.jsonl
├── checkpoints/
│   └── ljspeech_lora/       # LoRA adapter output
├── .env                     # WANDB_API_KEY (do not commit)
├── PLAN.md
└── TUTORIAL.md
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM during training | Reduce `per_device_train_batch_size` to 2 or 1; increase `gradient_accumulation_steps` proportionally |
| CUDA OOM during preprocessing | Add `--device cpu` (slower but no VRAM needed) |
| `ImportError: datasets` | Run `uv pip install datasets` |
| `ImportError: vinorm` | Run `uv pip install vinorm` |
| HuggingFace download hangs | Set `HF_DATASETS_OFFLINE=0` or use `--hf_split train` explicitly |
| `KeyError: bicodec_semantic_...` | Tokenizer doesn't know the special tokens — ensure you load from `model_dir/LLM`, not a generic Qwen2.5 checkpoint |
| WandB not logging | Check `.env` has the correct key; run `wandb login` as a fallback |
| `eval_strategy` deprecation warning | Upgrade `transformers`: `uv pip install -U transformers` |
