# SparkTTS Vietnamese Fine-tuning

Fine-tune SparkTTS (Qwen2.5-0.5B LLM + BiCodec audio tokenizer) on Vietnamese TTS data.

## Quick Start

```bash
# 1. Install training dependencies
uv pip install -r training/requirements_train.txt

# 2. Copy and fill in your WandB key
cp .env.example .env   # then edit WANDB_API_KEY

# 3. Prepare data (Phase 1 — small dataset for pipeline test)
python training/data/prepare_data.py hf --phase 1 --output_dir data/phase1

# 4. Pre-compute BiCodec tokens (run once, cached to disk)
python training/data/pretokenize.py \
    --metadata data/phase1/metadata.jsonl \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --cache_dir data/phase1/tokens

# 5. Train (LoRA fine-tuning, ~6h dataset, A100)
python training/train.py --config training/configs/train_phase1.yaml

# 6. Evaluate (UTMOSv2 MOS + WER)
python training/eval.py \
    --config training/configs/train_phase1.yaml \
    --checkpoint checkpoints/phase1/checkpoint-5000
```

---

## Directory Structure

```
training/
├── README.md                   ← this file
├── requirements_train.txt      ← pip/uv dependencies
├── train.py                    ← main training script (LoRA + WandB)
├── eval.py                     ← evaluation (UTMOSv2 + WER)
├── configs/
│   ├── train_phase1.yaml       ← Phase 1: ngochuyen_voice (~6h, 1 speaker)
│   └── train_phase2.yaml       ← Phase 2: VieNeu-TTS-140h (140h, 193 speakers)
└── data/
    ├── README.md               ← data pipeline documentation
    ├── prepare_data.py         ← download / convert → metadata.jsonl + WAVs
    ├── pretokenize.py          ← audio → BiCodec token cache (.pt files)
    ├── dataset.py              ← PyTorch Dataset + collate_fn
    └── text_processing.py      ← optional Vietnamese text normalization (pyvinorm)
```

---

## Architecture

SparkTTS has two components:

| Component | Description | Status during fine-tune |
|-----------|-------------|------------------------|
| **BiCodec** | Audio tokenizer: Wav2Vec2-XLSR-53 (53-lang) + VQ | **Frozen** — already multilingual |
| **LLM** (Qwen2.5-0.5B) | Text + speaker tokens → semantic tokens | **Fine-tuned** with LoRA |

The BiCodec is frozen because Wav2Vec2-large-XLSR-53 already supports Vietnamese (trained on 53 languages). Only the LLM needs to learn Vietnamese text → audio token mapping.

### Training Sequence

```
<|task_tts|><|start_content|>{vietnamese text}<|end_content|>
<|start_global_token|><|bicodec_global_0|>…<|bicodec_global_31|><|end_global_token|>
<|start_semantic_token|>  ← LLM predicts from here →  <|bicodec_semantic_X|>…<eos>
```

Loss is computed only on the semantic token positions.

---

## Datasets

| Phase | Dataset (HuggingFace) | Size | Speakers | Purpose |
|-------|-----------------------|------|----------|---------|
| 1 | `pnnbao-ump/ngochuyen_voice` | ~6h | 1 (F) | Pipeline test |
| 2 | `pnnbao-ump/VieNeu-TTS-140h` | 140h | 193 | Scale up |

Local LJSpeech-format datasets are also supported — see [data/README.md](data/README.md).

---

## Training (LoRA)

`train.py` fine-tunes only the LLM with LoRA adapters:

- **LoRA**: r=16, alpha=32, target modules: `q_proj k_proj v_proj o_proj`
- **Optimizer**: AdamW, lr=2e-4, cosine schedule, 100 warmup steps
- **Batch**: 8 per device × 4 grad accumulation = effective 32
- **Precision**: bf16 (A100 native)
- **Logging**: WandB (loss, lr, grad_norm per step; UTMOSv2 + WER at eval checkpoints)

```bash
python training/train.py --config training/configs/train_phase1.yaml

# Resume from checkpoint
python training/train.py --config training/configs/train_phase1.yaml \
    --resume_from checkpoints/phase1/checkpoint-2500
```

---

## Evaluation

`eval.py` generates speech for a held-out test set and computes two metrics:

| Metric | Tool | What it measures |
|--------|------|-----------------|
| **UTMOSv2 MOS** | [sarulab-speech/UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | Naturalness (1–5 float) |
| **WER** | Whisper large-v3 + jiwer | Intelligibility (lower is better) |

```bash
python training/eval.py \
    --config training/configs/train_phase1.yaml \
    --checkpoint checkpoints/phase1/checkpoint-5000 \
    --output eval_results.csv
```

Outputs `eval_results.csv` with per-sample scores and aggregate means printed to stdout.

---

## Environment Setup

```bash
# Install uv (fast package manager) if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt                  # base inference deps
uv pip install -r training/requirements_train.txt  # training deps
```

### WandB Configuration

```bash
cp .env.example .env
# Edit .env:
#   WANDB_API_KEY=your_key
#   WANDB_PROJECT=sparktts-vi
#   WANDB_ENTITY=your_entity
```

---

## Code Switching (EN ↔ VI)

No special handling is needed. Qwen2.5-0.5B uses a multilingual BPE tokenizer that
handles both English and Vietnamese natively. Mixed-language utterances (e.g.,
"Tôi dùng machine learning để phân tích") are processed as-is.

If pronunciation quality for English words in Vietnamese context is insufficient after
Phase 2 training, consider adding phonemization as a pre-processing step.
