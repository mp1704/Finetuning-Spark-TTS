# ASR Module — Issue Report

Issues encountered during initial setup and their resolutions.

---

## Issue 1: Eager import of MicrophoneStream crashes unrelated commands

**Error**
```
OSError: PortAudio library not found
```
Triggered by `uv run python -m asr.download_model` even though download has nothing to do with microphone capture.

**Root cause**
`asr/__init__.py` imported `MicrophoneStream` at the top level. `streaming.py` imports `sounddevice` at module load time. `sounddevice` immediately tries to locate the PortAudio shared library — and raises `OSError` if it's absent, even before any function is called.

**Fix** — `asr/__init__.py`
```python
# Before
from asr.streaming import MicrophoneStream

# After
try:
    from asr.streaming import MicrophoneStream
except OSError:
    MicrophoneStream = None  # PortAudio not installed; file/array modes still work
```
This makes PortAudio optional: file transcription and array transcription work without it.

---

## Issue 2: `from_transducer()` missing required `tokens` argument

**Error**
```
TypeError: OnlineRecognizer.from_transducer() missing 1 required positional argument: 'tokens'
```

**Root cause**
The HF repo `hynt/Zipformer-30M-RNNT-6000h` ships `bpe.model` but **no `tokens.txt`**.
The initial implementation assumed `bpe_vocab` alone was sufficient and omitted `tokens`.
Sherpa-ONNX always requires the `tokens` positional argument regardless of BPE usage.

**Fix — two-part**

1. `asr/download_model.py`: auto-generate `tokens.txt` from `bpe.model` after download using sentencepiece:
```python
sp = spm.SentencePieceProcessor()
sp.Load("bpe.model")
with open("tokens.txt", "w", encoding="utf-8") as f:
    for i in range(sp.GetPieceSize()):
        f.write(f"{sp.IdToPiece(i)} {i}\n")
```

2. `asr/recognizer.py`: pass `tokens=str(config.tokens_path)` to `from_transducer()`.

3. `asr/requirements.txt`: add `sentencepiece>=0.1.99`.

---

## Issue 3: `'encoder_dims' does not exist in the metadata`

**Error**
```
/project/sherpa-onnx/csrc/online-zipformer2-transducer-model.cc:InitEncoder:112
'encoder_dims' does not exist in the metadata
```

**Root cause**
Inspecting the ONNX encoder with the `onnx` library revealed the actual model metadata:
```python
import onnx
m = onnx.load("encoder-epoch-20-avg-10.int8.onnx")
print({p.key: p.value for p in m.metadata_props})
# {'model_type': 'zipformer2', 'comment': 'non-streaming zipformer2', ...}
```

The comment `non-streaming zipformer2` is the critical finding. Despite the HF repo being named "Zipformer-30M-RNNT" (implying streaming), the ONNX was exported as a **non-streaming (offline/batch) model**.

`OnlineRecognizer` (streaming API) expects the streaming Zipformer2 encoder format, which embeds `encoder_dims` in the ONNX metadata. The non-streaming encoder does not have this field, causing the crash.

Attempts to downgrade sherpa-onnx to older versions (1.9.28, 1.10.38, 1.8.0) failed — those versions are not available for Python 3.11 on the current platform.

**Fix** — `asr/recognizer.py`: replace `OnlineRecognizer` with `OfflineRecognizer`
```python
# Before
self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=..., encoder=..., decoder=..., joiner=...,
    bpe_vocab=..., sample_rate=..., feature_dim=...,
    decoding_method=..., max_active_paths=...,
)

# After
self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    tokens=..., encoder=..., decoder=..., joiner=...,
    num_threads=..., provider=...,
)
```

**Impact on pipeline**
- `process_file()` and `process_array()` — unchanged behaviour.
- `process_stream()` — no longer provides incremental partial results (not possible with an offline model). The VAD still detects utterance boundaries in real time; the ASR transcribes each completed utterance as a whole. For the VAD → ASR → LLM → TTS use case this is sufficient: the LLM is only invoked after a complete utterance is detected anyway.
- Removed `feature_dim`, `decoding_method`, `max_active_paths` from `ASRConfig` (not used by `OfflineRecognizer`).
- Removed streaming helper methods `create_stream()`, `feed_chunk()`, `finalize_stream()` from `ZipformerRecognizer` (replaced by single `transcribe()` call).

---

## Issue 4: `soundfile.LibsndfileError: Format not recognised`

**Error**
```
soundfile.LibsndfileError: Error opening 'ref2.wav': Format not recognised.
```

**Root cause**
`soundfile` wraps `libsndfile` which handles standard PCM-encoded formats (WAV/FLAC/OGG/AIFF). It does **not** support MP3, AAC, or WAV files with non-standard encodings (e.g. WAV-MP3, RF64, extensible-format WAV with unusual codecs).

**Fix** — `asr/pipeline.py`: try soundfile first, fall back to `torchaudio`
```python
try:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
except Exception:
    import torchaudio
    waveform, sr = torchaudio.load(str(path))   # handles MP3, AAC, etc.
    audio = waveform.mean(dim=0).numpy()        # mix to mono
```

`torchaudio` is already a project dependency (used by the TTS components) and supports a much wider range of codecs via FFmpeg/sox backends.

---

## Summary of file changes

| File | Change |
|---|---|
| `__init__.py` | Lazy import of `MicrophoneStream` with `try/except OSError` |
| `config.py` | Added `tokens_path` property; removed `feature_dim`, `decoding_method`, `max_active_paths` |
| `recognizer.py` | Switched `OnlineRecognizer` → `OfflineRecognizer`; simplified to single `transcribe()` method |
| `pipeline.py` | Simplified `process_stream()`; added torchaudio fallback in `_load_audio()` |
| `download_model.py` | Auto-generate `tokens.txt` from `bpe.model` via sentencepiece |
| `requirements.txt` | Added `sentencepiece>=0.1.99` |
| `PLAN.md` | Updated to reflect offline model and corrected API descriptions |
