# Spark-TTS Fine-tuning

LoRA fine-tuning and inference for [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) on custom voice datasets.

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone <this-repo>
cd spa1

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### WandB

Training logs to [Weights & Biases](https://wandb.ai). Create a `.env` file in the project root:

```sh
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=sparktts-finetune
WANDB_ENTITY=your_wandb_username
```

The training script loads `.env` automatically via `python-dotenv`. The project and run name can also be set per-run in `finetune/config.yaml` under the `wandb:` key.

---

## 1. Download Pretrained Model

```sh
bash script/00_download_pretrain.sh
```

Downloads `SparkAudio/Spark-TTS-0.5B` into `pretrained_models/Spark-TTS-0.5B/`.

---

## 2. Prepare Dataset

### Option A — LJSpeech (English)

Download and split LJSpeech:

```sh
bash download_ljspeech.sh
```

This downloads, extracts to `LJSpeech-1.1/`, and creates `metadata_train.csv` / `metadata_val.csv` splits.

Then preprocess (tokenize audio → JSONL):

```sh
CUDA_VISIBLE_DEVICES=0 uv run python -m finetune.preprocess \
    --source ljspeech \
    --data_dir LJSpeech-1.1 \
    --output_dir data/ljspeech \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --device cuda
```

Output: `data/ljspeech/train_tokens.jsonl` and `data/ljspeech/val_tokens.jsonl`.

---

### Option B — HuggingFace Dataset (e.g. Vietnamese voice)

```sh
CUDA_VISIBLE_DEVICES=0 uv run python -m finetune.preprocess \
    --source huggingface \
    --hf_dataset <your-hf-dataset-id> \
    --hf_audio_col audio \
    --hf_text_col transcription \
    --hf_id_col file_name \
    --text_normalizer vinorm \
    --output_dir data/myvoice \
    --val_size 500 \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --device cuda
```

**HuggingFace preprocess arguments:**

| Flag | Description |
|------|-------------|
| `--hf_dataset` | HuggingFace dataset ID (e.g. `pnnbao-ump/ngochuyen_voice`) |
| `--hf_audio_col` | Column name containing audio |
| `--hf_text_col` | Column name containing text transcription |
| `--hf_id_col` | Column name for sample ID (optional) |
| `--text_normalizer` | `none` (default) or `vinorm` for Vietnamese |
| `--val_size` | Number of samples held out for validation |
| `--output_dir` | Where to write `train_tokens.jsonl` / `val_tokens.jsonl` |

Output: `data/myvoice/train_tokens.jsonl` and `data/myvoice/val_tokens.jsonl`.

---

## 3. Configure Training

Edit [finetune/config.yaml](finetune/config.yaml) — key fields to change:

```yaml
model_dir: pretrained_models/Spark-TTS-0.5B

data:
  train_jsonl: data/myvoice/train_tokens.jsonl
  val_jsonl:   data/myvoice/val_tokens.jsonl

training:
  output_dir: checkpoints/myvoice
  num_train_epochs: 6
  # resume_from_checkpoint: checkpoints/myvoice/checkpoint-660  # see "Resume" below

wandb:
  project: sparktts-finetune   # overrides WANDB_PROJECT from .env
  run_name: myvoice-lora
```

---

## 4. Train

```sh
CUDA_VISIBLE_DEVICES=0 uv run python -m finetune.train --config finetune/config.yaml
```

Checkpoints are saved to `training.output_dir` every `save_steps` steps. The final LoRA adapter is written to `output_dir/` on completion.

### Resume from a checkpoint

1. Increase `num_train_epochs` to the new total.
2. Set `resume_from_checkpoint` to the latest checkpoint directory:

```yaml
training:
  num_train_epochs: 6
  resume_from_checkpoint: checkpoints/myvoice/checkpoint-660
```

3. Run training again. The Trainer restores optimizer state, scheduler, and RNG from the checkpoint and continues until the total epoch count is reached.

---

## 5. Inference

### Voice cloning mode (use a reference audio)

```sh
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --adapter_dir checkpoints/myvoice \
    --text "Text to synthesize." \
    --prompt_speech_path path/to/reference.wav \
    --output_path outputs/result.wav
```

### Controllable mode (no reference audio)

```sh
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --adapter_dir checkpoints/myvoice \
    --text "Text to synthesize." \
    --gender female \
    --pitch moderate \
    --speed moderate \
    --output_path outputs/result.wav
```

### Vietnamese text (with normalizer)

```sh
CUDA_VISIBLE_DEVICES=0 uv run python inference/run_finetuned_inference.py \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --adapter_dir checkpoints/myvoice \
    --text "Ngày 25/12/2024, nhiệt độ đạt 32°C." \
    --gender female \
    --text_normalizer vinorm \
    --output_path outputs/result.wav
```

**All inference arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | `pretrained_models/Spark-TTS-0.5B` | Base model directory |
| `--adapter_dir` | required | LoRA adapter directory |
| `--text` | required | Text to synthesize |
| `--prompt_speech_path` | — | Reference wav for voice cloning mode |
| `--prompt_text` | — | Transcript of reference audio (optional) |
| `--gender` | — | `male` or `female` — enables controllable mode |
| `--pitch` | `moderate` | `very_low / low / moderate / high / very_high` |
| `--speed` | `moderate` | `very_low / low / moderate / high / very_high` |
| `--text_normalizer` | `none` | `none` or `vinorm` |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_k` | `50` | Top-k sampling |
| `--top_p` | `0.95` | Top-p sampling |
| `--device` | `cuda` | `cuda`, `cuda:0`, or `cpu` |
| `--output_path` | `outputs/output_finetuned.wav` | Output wav path |

> Provide either `--prompt_speech_path` (voice cloning) or `--gender` (controllable), not both.

---

## Project Structure

```
finetune/
  config.yaml          # training configuration
  preprocess.py        # tokenize dataset → JSONL
  train.py             # LoRA training with HuggingFace Trainer
  dataset.py           # dataset loader
inference/
  run_finetuned_inference.py   # inference with LoRA adapter
script/
  00_download_pretrain.sh      # download base model
  01_pre.sh                    # example preprocess commands
  02_train.sh                  # run training
  03_inference.sh              # example inference commands
pretrained_models/
  Spark-TTS-0.5B/              # base model (downloaded in step 1)
checkpoints/
  <run-name>/                  # LoRA adapter + intermediate checkpoints
data/
  <dataset>/
    train_tokens.jsonl
    val_tokens.jsonl
```

---

## Citation

```bibtex
@misc{wang2025sparktts,
  title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens},
  author={Xinsheng Wang et al.},
  year={2025},
  eprint={2503.01710},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2503.01710}
}
```
