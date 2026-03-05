# ASR Module — Architecture & Design

## Overview

This module implements the **VAD → ASR** stage of the conversational pipeline:

```
[Audio Input] → VAD → ASR → [Text to LLM] → [TTS speaks back]
```

The ASR module is fully self-contained. It exposes `TranscriptionResult` objects that downstream LLM/TTS stages consume without needing any audio knowledge.

---

## Model Choice: hynt/Zipformer-30M-RNNT-6000h

| Property | Value |
|---|---|
| Architecture | Zipformer2 + RNNT (transducer) |
| Mode | **Non-streaming** (offline batch) |
| Training data | ~6000h Vietnamese speech |
| WER (VLSP2025) | 7.97% public / 8.10% private |
| Latency (CPU, 12s audio) | ~0.3s |
| Latency (GPU RTX 3090) | <0.1s |
| Tokenizer | BPE (SentencePiece `bpe.model`) |
| ONNX metadata | `model_type: zipformer2, comment: non-streaming zipformer2` |

**Why Zipformer RNNT?**
- Low parameter count (30M) → fast on CPU
- Best-in-class WER for Vietnamese in open weights
- ONNX export → no k2 dependency at inference time

**Why Sherpa-ONNX?**
- Production-ready Python API for ONNX transducer models
- Handles feature extraction (80-dim log-filterbank) internally
- `OfflineRecognizer` correctly handles the non-streaming model format

> **Note:** The HF repo describes this as "Zipformer-30M-RNNT" but the ONNX metadata
> confirms it is a **non-streaming Zipformer2** model. `OnlineRecognizer` (streaming API)
> raises `'encoder_dims' does not exist in the metadata`. Use `OfflineRecognizer` instead.
> See `REPORT.md` for the full investigation.

---

## VAD Choice: Silero VAD

**Why Silero VAD?**
- State-of-the-art accuracy for speech/non-speech classification
- Very fast (runs in <1ms per 32ms frame on CPU)
- PyTorch JIT model via `torch.hub` — no extra ONNX runtime conflict with sherpa-onnx

**Hard constraint:** Silero VAD accepts exactly **512 samples at 16 kHz** per frame.
The entire pipeline is designed around this: `blocksize=512` in sounddevice, `vad_chunk_size=512` in config.

---

## Component Design

```
┌─────────────────────────────────────────────────────────┐
│                      ASRPipeline                        │
│  ┌──────────┐   SpeechSegment   ┌───────────────────┐  │
│  │SileroVAD │ ───────────────►  │ZipformerRecognizer│  │
│  │(512-samp │                   │OfflineRecognizer  │  │
│  │ chunks)  │                   │(sherpa_onnx)      │  │
│  └──────────┘                   └───────────────────┘  │
│        ▲                                 │              │
│        │                                 ▼              │
│   process_chunk()               TranscriptionResult     │
│   (512 samples)                    (.text, .segment)    │
└─────────────────────────────────────────────────────────┘
```

### `ASRConfig` (config.py)
Single source of truth. Key defaults:
- `use_int8=True`: int8-quantised ONNX (encoder 27.7 MB vs 92.2 MB full)
- `provider="cpu"`: switch to `"cuda"` for GPU
- `vad_chunk_size=512`: fixed, do not change

### `SileroVAD` (vad.py)
Stateful chunk processor. State machine:
- `_triggered=False`: looking for speech onset
- `_triggered=True`: accumulating speech, waiting for silence to end utterance
- Emits `SpeechSegment` when silence duration ≥ `min_silence_ms`
- `flush()` captures trailing speech at end-of-file

### `ZipformerRecognizer` (recognizer.py)
Thin wrapper around `sherpa_onnx.OfflineRecognizer`.
- Uses `tokens.txt` (generated from `bpe.model` by `download_model.py` via sentencepiece)
- `transcribe(samples)` — creates a new stream per call, decodes whole segment at once

### `ASRPipeline` (pipeline.py)
Orchestrates VAD + ASR. Three entry points:
1. `process_file(path)` — audio file → `List[TranscriptionResult]`
2. `process_array(audio, sr)` — numpy array → same (LLM integration point)
3. `process_stream(chunk_iter)` — VAD accumulates chunks; transcribes each completed segment whole

### `MicrophoneStream` (streaming.py)
Separates audio capture (sounddevice audio thread) from inference (background Python thread)
via `queue.Queue`. The `blocksize=512` in `sd.InputStream` guarantees exactly 512 samples
per callback — no buffering logic needed. Results arrive per utterance (after silence gap),
not incrementally (because the ASR model is offline/non-streaming).

---

## Int8 Quantisation Decision

| File | Size | Speed (CPU) |
|---|---|---|
| `encoder-epoch-20-avg-10.onnx` | 92.2 MB | baseline |
| `encoder-epoch-20-avg-10.int8.onnx` | 27.7 MB | ~2× faster |

Int8 is the default (`use_int8=True`): smaller download, faster CPU inference, no accuracy loss.

---

## tokens.txt Generation

The HF repo ships `bpe.model` but no `tokens.txt`. `sherpa_onnx.OfflineRecognizer.from_transducer()`
requires a `tokens` argument. `download_model.py` auto-generates `tokens.txt` using sentencepiece:

```python
sp = spm.SentencePieceProcessor()
sp.Load("bpe.model")
with open("tokens.txt", "w") as f:
    for i in range(sp.GetPieceSize()):
        f.write(f"{sp.IdToPiece(i)} {i}\n")
```

---

## LLM/TTS Integration

```python
from asr import ASRPipeline

pipeline = ASRPipeline()

# Option 1: file (batch)
for result in pipeline.process_file("turn.wav"):
    if not result.is_empty:
        response = llm.generate(result.text)
        tts.speak(response)

# Option 2: array (from a WebRTC/PyAudio buffer)
results = pipeline.process_array(audio_np, sample_rate=48000)

# Option 3: microphone streaming (results arrive per completed utterance)
from asr import MicrophoneStream
import threading

def on_utterance(result):
    threading.Thread(
        target=lambda: tts.speak(llm.generate(result.text))
    ).start()

with MicrophoneStream(pipeline) as mic:
    mic.start(on_result=on_utterance)
    input("Listening... press Enter to stop\n")
```

`TranscriptionResult.segment` carries `start_seconds`, `end_seconds`, `duration_seconds`
for turn-taking heuristics in the conversation manager.
