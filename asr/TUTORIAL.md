# ASR Pipeline — Tutorial

Vietnamese ASR using Zipformer-30M-RNNT (7.97% WER, ~0.3s/12s audio on CPU).

---

## Prerequisites

- Python 3.10+
- `uv` installed (`pip install uv`)
- Project virtual environment already set up (`uv venv`)

---

## Step 1: Install ASR dependencies

```bash
uv pip install -r asr/requirements.txt
```

On Linux, microphone streaming requires PortAudio. Build it locally (no sudo needed):
```bash
curl -LO https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
tar xf pa_stable_v190700_20210406.tgz && cd portaudio
./configure --prefix=$HOME/.local && make -j$(nproc) && make install
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

Add the export to your shell profile to make it permanent:
```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## Step 2: Download the model

Downloads int8-quantised ONNX files (~30 MB total) from HuggingFace:

```bash
uv run python -m asr.download_model
```

Files are saved to `asr/models/zipformer-30m/`. To also download full-precision ONNX:

```bash
uv run python -m asr.download_model --full
```

Custom location:
```bash
uv run python -m asr.download_model --model-dir /path/to/my/model
```

---

## Step 3: Transcribe an audio file

```bash
uv run python -m asr.transcribe audio.wav
```

Output: one line per detected speech segment (after VAD filtering).

---

## Step 4: Show timestamps

```bash
uv run python -m asr.transcribe audio.wav --show-timestamps
```

Example output:
```
[0.32s – 4.15s]  xin chào tôi muốn đặt một cuộc hẹn
[5.80s – 9.22s]  vào ngày mai lúc mười giờ sáng
```

---

## Step 5: Save transcription to file

```bash
uv run python -m asr.transcribe audio.wav -o output.txt
uv run python -m asr.transcribe audio.wav --show-timestamps -o output.txt
```

---

## Step 6: GPU acceleration

```bash
uv run python -m asr.transcribe audio.wav --provider cuda
```

Or in Python:
```python
from asr import ASRConfig, ASRPipeline

pipeline = ASRPipeline(ASRConfig(provider="cuda"))
```

---

## Step 7: Python API — process a file

```python
from asr import ASRPipeline

pipeline = ASRPipeline()

results = pipeline.process_file("audio.wav")
for r in results:
    print(f"[{r.segment.start_seconds:.2f}s] {r.text}")
```

---

## Step 8: Python API — process a numpy array

Useful when audio comes from another component (no file I/O needed):

```python
import numpy as np
from asr import ASRPipeline

pipeline = ASRPipeline()

# audio: float32, any sample rate — will be resampled to 16 kHz automatically
audio = np.zeros(48000, dtype=np.float32)   # replace with real audio
results = pipeline.process_array(audio, sample_rate=48000)
```

---

## Step 9: Real-time microphone streaming

```python
from asr import ASRPipeline, MicrophoneStream

pipeline = ASRPipeline()

def on_utterance(result):
    print(f"Heard: {result.text}")
    # → pass result.text to your LLM here

with MicrophoneStream(pipeline) as mic:
    mic.start(on_result=on_utterance)
    input("Listening... press Enter to stop\n")
```

---

## Step 10: Full conversational loop (VAD → ASR → LLM → TTS)

```python
import threading
from asr import ASRPipeline, MicrophoneStream

# Assume llm and tts are already initialised elsewhere in the project
pipeline = ASRPipeline()

def handle_utterance(result):
    if result.is_empty:
        return
    # Run LLM + TTS in background so ASR keeps capturing
    def respond():
        response = llm.generate(result.text)
        tts.speak(response)
    threading.Thread(target=respond, daemon=True).start()

with MicrophoneStream(pipeline) as mic:
    mic.start(on_result=handle_utterance)
    input("Press Enter to stop...\n")
```

---

## Configuration reference

```python
from asr import ASRConfig, ASRPipeline

config = ASRConfig(
    model_dir="asr/models/zipformer-30m",  # path to ONNX files
    use_int8=True,                          # int8 quantised (recommended)
    provider="cpu",                         # "cpu" | "cuda" | "coreml"
    num_threads=4,                          # CPU inference threads
    vad_threshold=0.5,                      # 0.0–1.0, higher = less sensitive
    vad_min_speech_ms=250,                  # ignore speech shorter than this
    vad_min_silence_ms=300,                 # silence gap to split utterances
    vad_speech_pad_ms=100,                  # padding added around speech
    stream_device=None,                     # None = default mic
)
pipeline = ASRPipeline(config)
```

---

## Troubleshooting

**`FileNotFoundError: Missing encoder: asr/models/zipformer-30m/...`**
→ Run `uv run python -m asr.download_model` first.

**`ImportError: sherpa_onnx is not installed`**
→ Run `uv pip install sherpa-onnx`

**`ImportError: sounddevice is not installed`**
→ Run `uv pip install sounddevice` + `sudo apt-get install portaudio19-dev` (Linux)

**Empty transcription on a file with speech**
→ Lower `vad_threshold` (e.g. `ASRConfig(vad_threshold=0.3)`) or reduce `vad_min_silence_ms`.

**Wrong sample rate warning**
→ The pipeline auto-resamples to 16 kHz using soxr — no action needed.

**CUDA `provider` fails**
→ Install sherpa-onnx with CUDA support: `uv pip install sherpa-onnx-cuda`
   or fall back to `provider="cpu"`.
