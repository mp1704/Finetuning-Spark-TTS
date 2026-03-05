from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

from asr.config import ASRConfig
from asr.vad import SileroVAD, SpeechSegment
from asr.recognizer import ZipformerRecognizer


@dataclass
class TranscriptionResult:
    """Output of the ASR pipeline for one detected speech utterance."""

    text: str               # ASR transcript (Vietnamese)
    segment: SpeechSegment  # VAD segment containing audio + timing

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def __str__(self) -> str:
        return (
            f"[{self.segment.start_seconds:.2f}s – {self.segment.end_seconds:.2f}s] "
            f"{self.text}"
        )


# Callback type for streaming results
TranscriptionCallback = Callable[[TranscriptionResult], None]


class ASRPipeline:
    """
    End-to-end VAD → ASR pipeline.

    Supports three modes:
      - process_file(path)    : transcribe an audio file
      - process_array(audio)  : transcribe a numpy array (e.g. from another pipeline stage)
      - process_stream(iter)  : real-time chunk-by-chunk transcription

    Downstream integration (LLM/TTS):
        for result in pipeline.process_file("turn.wav"):
            response = llm.generate(result.text)
            tts.speak(response)
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.vad = SileroVAD(
            threshold=self.config.vad_threshold,
            min_speech_ms=self.config.vad_min_speech_ms,
            min_silence_ms=self.config.vad_min_silence_ms,
            speech_pad_ms=self.config.vad_speech_pad_ms,
            sample_rate=self.config.sample_rate,
        )
        self.recognizer = ZipformerRecognizer(self.config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_audio(self, path: Path) -> np.ndarray:
        """Load audio file, mix to mono, resample to 16 kHz."""
        try:
            import soxr
        except ImportError as e:
            raise ImportError("uv pip install soxr") from e

        # Try soundfile first (fast, handles standard PCM WAV/FLAC/OGG)
        # Fall back to torchaudio which handles MP3, AAC, non-standard WAV, etc.
        audio = None
        sr = None
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            import torch
            import torchaudio
            waveform, sr = torchaudio.load(str(path))   # (C, T) float32
            audio = waveform.mean(dim=0).numpy()        # mix to mono

        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.config.sample_rate:
            audio = soxr.resample(audio, sr, self.config.sample_rate, quality="VHQ")
        return audio.astype(np.float32)

    def _results_from_segments(
        self, segments: List[SpeechSegment]
    ) -> List[TranscriptionResult]:
        return [
            TranscriptionResult(text=self.recognizer.transcribe(seg.samples), segment=seg)
            for seg in segments
        ]

    # ------------------------------------------------------------------
    # File mode
    # ------------------------------------------------------------------

    def process_file(self, audio_path: Path | str) -> List[TranscriptionResult]:
        """
        Load an audio file and transcribe all speech segments.

        Accepts WAV, FLAC, OGG, and other formats supported by soundfile.
        Resamples to 16 kHz automatically.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of TranscriptionResult objects (one per VAD-detected segment).
            Returns an empty list if no speech is detected.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio = self._load_audio(path)
        segments = self.vad.process_audio(audio)
        if not segments:
            return []
        return self._results_from_segments(segments)

    # ------------------------------------------------------------------
    # Array mode
    # ------------------------------------------------------------------

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[TranscriptionResult]:
        """
        Transcribe audio from a numpy array.

        Useful when audio comes directly from another pipeline component
        (e.g. a WebRTC buffer or a PyAudio capture) without writing to disk.

        Args:
            audio: float32 numpy array, mono or stereo
            sample_rate: sample rate of the input audio

        Returns:
            List of TranscriptionResult objects.
        """
        audio = audio.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sample_rate != self.config.sample_rate:
            try:
                import soxr
            except ImportError as e:
                raise ImportError("uv pip install soxr") from e
            audio = soxr.resample(audio, sample_rate, self.config.sample_rate, quality="VHQ")

        segments = self.vad.process_audio(audio)
        if not segments:
            return []
        return self._results_from_segments(segments)

    # ------------------------------------------------------------------
    # Streaming mode
    # ------------------------------------------------------------------

    def process_stream(
        self,
        chunk_iter: Iterator[np.ndarray],
        on_result: Optional[TranscriptionCallback] = None,
    ) -> Iterator[TranscriptionResult]:
        """
        Process a stream of audio chunks, yielding results as utterances complete.

        Each chunk must be exactly config.vad_chunk_size (512) samples.
        VAD accumulates chunks into speech segments; each completed segment is
        transcribed in full by the offline OfflineRecognizer.

        Args:
            chunk_iter: Iterator of float32 numpy arrays, each of shape (512,)
            on_result: Optional callback called for each result (in addition to yielding)

        Yields:
            TranscriptionResult for each completed utterance
        """
        self.vad.reset_state()

        for chunk in chunk_iter:
            completed = self.vad.process_chunk(chunk)
            for seg in completed:
                text = self.recognizer.transcribe(seg.samples)
                result = TranscriptionResult(text=text, segment=seg)
                if on_result:
                    on_result(result)
                yield result

        # Drain any trailing speech at end-of-stream
        for seg in self.vad.flush():
            text = self.recognizer.transcribe(seg.samples)
            result = TranscriptionResult(text=text, segment=seg)
            if on_result:
                on_result(result)
            yield result
