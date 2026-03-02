# Spark-TTS Vietnamese Fine-tuning

This repo builds on [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) — all inference code is unchanged. We add a LoRA training and evaluation pipeline for Vietnamese TTS (`training/` directory).

> **Original paper:** [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](https://arxiv.org/pdf/2503.01710)

---

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install training dependencies
uv sync --extra train

# Configure WandB
cp .env.example .env   # edit WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
```

Download the base model:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')
"
```

---

## Training Pipeline

```bash
# 1. Prepare data — download HF dataset, write wavs + metadata.jsonl
#    --normalize applies Vietnamese text normalization (pyvinorm)
uv run python training/data/prepare_data.py hf --phase 1 --output_dir data/phase1 --normalize

# LJSpeech-format local directory also supported:
# uv run python training/data/prepare_data.py ljspeech \
#     --ljspeech_dir /path/to/dataset --output_dir data/custom --speaker_id myspeaker

# 2. Pre-compute BiCodec tokens (run once, cached to disk)
CUDA_VISIBLE_DEVICES=0 uv run python training/data/pretokenize.py \
    --metadata data/phase1/metadata.jsonl \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --cache_dir data/phase1/tokens \
    --device cuda:0

# 3. Train (LoRA fine-tuning, WandB logging)
CUDA_VISIBLE_DEVICES=0 uv run python training/train.py --config training/configs/train_phase1.yaml

# 4. Evaluate (UTMOSv2 MOS + Whisper WER)
uv run python training/eval.py \
    --config training/configs/train_phase1.yaml \
    --checkpoint checkpoints/phase1
```

---

## Inference with Fine-tuned Checkpoint

First, merge LoRA weights into a standalone model directory (one-time step):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained('pretrained_models/Spark-TTS-0.5B/LLM', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, 'checkpoints/phase1')
model = model.merge_and_unload()
model.save_pretrained('pretrained_models/Spark-TTS-0.5B-vi/LLM')
AutoTokenizer.from_pretrained('pretrained_models/Spark-TTS-0.5B/LLM').save_pretrained('pretrained_models/Spark-TTS-0.5B-vi/LLM')
"
mkdir -p pretrained_models/Spark-TTS-0.5B-vi
ln -sf $(pwd)/pretrained_models/Spark-TTS-0.5B/BiCodec pretrained_models/Spark-TTS-0.5B-vi/BiCodec
ln -sf $(pwd)/pretrained_models/Spark-TTS-0.5B/wav2vec2-large-xlsr-53 pretrained_models/Spark-TTS-0.5B-vi/wav2vec2-large-xlsr-53
cp pretrained_models/Spark-TTS-0.5B/config.yaml pretrained_models/Spark-TTS-0.5B-vi/config.yaml
```

Then run inference:

```bash
# Voice cloning — provide a reference audio
CUDA_VISIBLE_DEVICES=0 python -m cli.inference \
    --text "Xin chào, tôi là trợ lý giọng nói." \
    --device 0 \
    --save_dir example/results \
    --model_dir pretrained_models/Spark-TTS-0.5B-vi \
    --prompt_text "transcript of the reference audio" \
    --prompt_speech_path path/to/reference.wav

# Controlled voice — no reference audio needed
CUDA_VISIBLE_DEVICES=0 python -m cli.inference \
    --text "Xin chào, tôi là trợ lý giọng nói." \
    --device 0 \
    --save_dir example/results \
    --model_dir pretrained_models/Spark-TTS-0.5B-vi \
    --gender male --pitch moderate --speed moderate
```

Or use the example script:
```bash
bash example/infer_vi.sh
```

---

## Datasets

| Phase | Dataset | Size | Purpose |
|-------|---------|------|---------|
| 1 | `pnnbao-ump/ngochuyen_voice` | ~6h, 1 speaker | Pipeline validation |
| 2 | `pnnbao-ump/VieNeu-TTS-140h` | 140h, 193 speakers | Scale up |

---

## Citation

```
@misc{wang2025sparktts,
      title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens},
      author={Xinsheng Wang and Mingqi Jiang and Ziyang Ma and Ziyu Zhang and Songxiang Liu and Linqin Li and Zheng Liang and Qixi Zheng and Rui Wang and Xiaoqin Feng and Weizhen Bian and Zhen Ye and Sitong Cheng and Ruibin Yuan and Zhixian Zhao and Xinfa Zhu and Jiahao Pan and Liumeng Xue and Pengcheng Zhu and Yunlin Chen and Zhifei Li and Xie Chen and Lei Xie and Yike Guo and Wei Xue},
      year={2025},
      eprint={2503.01710},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.01710},
}
```
