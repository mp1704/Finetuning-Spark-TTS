"""
ASR package: VAD → Zipformer RNNT pipeline for Vietnamese speech recognition.

Model: hynt/Zipformer-30M-RNNT-6000h (7.97% WER, ~0.3s for 12s audio on CPU)
VAD:   Silero VAD
ASR:   Sherpa-ONNX (OnlineRecognizer, transducer/RNNT)

Quick start:
    from asr import ASRPipeline

    pipeline = ASRPipeline()
    for result in pipeline.process_file("audio.wav"):
        print(result.text)

Streaming microphone:
    from asr import ASRPipeline, MicrophoneStream

    with MicrophoneStream(ASRPipeline()) as mic:
        mic.start(on_result=lambda r: print(r.text))
        input("Press Enter to stop...")

Download model first:
    uv run python -m asr.download_model
"""

from asr.config import ASRConfig
from asr.vad import SileroVAD, SpeechSegment
from asr.recognizer import ZipformerRecognizer
from asr.pipeline import ASRPipeline, TranscriptionResult

# MicrophoneStream requires sounddevice + PortAudio — import lazily so the
# rest of the package works even when PortAudio is not installed.
try:
    from asr.streaming import MicrophoneStream
except OSError:
    MicrophoneStream = None  # type: ignore[assignment,misc]

__all__ = [
    "ASRConfig",
    "SileroVAD",
    "SpeechSegment",
    "ZipformerRecognizer",
    "ASRPipeline",
    "TranscriptionResult",
    "MicrophoneStream",
]
