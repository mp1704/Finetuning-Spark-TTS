# SparkTTS LJSpeech Finetuning — Implementation Plan

## Overview
Finetune the SparkTTS LLM on LJSpeech (single-speaker English TTS, ~24h) using LoRA. BiCodec and Wav2Vec2 are frozen throughout.

## Architecture
```
LLM (Qwen2.5)  ←── LoRA finetuned
BiCodec         ←── frozen (pretrained audio codec)
Wav2Vec2        ←── frozen (feature extractor, used only in preprocessing)
```

## Training Format
Each sample is formatted as a causal-LM sequence:
```
INPUT (no loss):   <|task_tts|><|start_content|>TEXT<|end_content|>
                   <|start_global_token|>GLOBAL_TOKENS<|end_global_token|>
                   <|start_semantic_token|>
TARGET (loss):     SEMANTIC_TOKENS + <eos>
```
Global tokens encode the speaker identity from the sample's own audio.
Semantic tokens encode the speech content the LLM must learn to generate.

## Files
| File | Purpose |
|------|---------|
| `finetune/preprocess.py` | One-time: tokenize all WAVs → JSONL |
| `finetune/dataset.py` | PyTorch Dataset + collate_fn |
| `finetune/train.py` | LoRA training with WandB logging |
| `finetune/config.yaml` | All hyperparameters |

## Reused Existing Code
| Existing file | Reused for |
|---------------|-----------|
| `sparktts/models/audio_tokenizer.py` | `BiCodecTokenizer.tokenize()` in preprocessing |
| `sparktts/utils/audio.py` | `load_audio()` inside BiCodecTokenizer |
| `sparktts/utils/token_parser.py` | `TASK_TOKEN_MAP` for prompt format |

## Dependencies (new)
```
peft           # LoRA
wandb          # experiment logging
python-dotenv  # load WANDB_API_KEY from .env
```
Install: `uv pip install peft wandb python-dotenv`

## Verification Steps
1. Preprocess: check JSONL line counts match CSV (12 000 train, 1 100 val)
2. Spot-check a JSONL record: semantic token count ≈ audio_seconds × 50
3. Short training run with `num_train_epochs: 0.05` — confirm loss decreases and WandB dashboard updates
4. Load adapter + run inference on a val prompt to assess voice quality
