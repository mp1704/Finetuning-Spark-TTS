from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class SpeechSegment:
    """A contiguous block of audio classified as speech by VAD."""

    samples: np.ndarray     # float32, shape (N,), range [-1.0, 1.0]
    start_sample: int       # offset from start of the original stream
    end_sample: int
    sample_rate: int = 16000

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


class SileroVAD:
    """
    Silero VAD wrapper for detecting speech in audio.

    Hard constraint from the Silero model:
        chunk_size = 512 samples at 16 kHz (exactly 32 ms per frame)

    Two usage patterns:
      - File mode: process_audio(full_array) → List[SpeechSegment]
      - Streaming mode: process_chunk(512_samples) in a loop, then flush()
    """

    CHUNK_SIZE = 512   # enforced by the Silero ONNX model

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)

        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,     # PyTorch JIT — avoids onnxruntime version conflicts
            verbose=False,
        )
        self._model.eval()
        self.reset_state()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset RNN hidden state and internal buffers. Call between files."""
        self._model.reset_states()
        self._triggered = False
        self._silence_samples = 0
        self._speech_start = 0
        self._speech_buffer: list[float] = []
        self._sample_offset = 0

    @property
    def is_triggered(self) -> bool:
        """True when VAD is currently inside a speech region."""
        return self._triggered

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _speech_prob(self, chunk: np.ndarray) -> float:
        tensor = torch.from_numpy(chunk).float().unsqueeze(0)   # (1, 512)
        with torch.no_grad():
            return self._model(tensor, self.sample_rate).item()

    # ------------------------------------------------------------------
    # Streaming mode (chunk-by-chunk)
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> List[SpeechSegment]:
        """
        Feed exactly CHUNK_SIZE (512) samples.

        Returns a list of completed SpeechSegments (0 or 1 per call).
        A segment is returned only when a silence gap terminates a speech region.

        Args:
            chunk: float32 numpy array of shape (512,)
        """
        if len(chunk) != self.CHUNK_SIZE:
            raise ValueError(
                f"SileroVAD requires exactly {self.CHUNK_SIZE} samples per chunk, "
                f"got {len(chunk)}"
            )

        completed: List[SpeechSegment] = []
        prob = self._speech_prob(chunk)

        if prob >= self.threshold:
            if not self._triggered:
                # Speech onset — record start with pre-roll padding
                self._triggered = True
                self._silence_samples = 0
                pad_start = max(0, self._sample_offset - self.speech_pad_samples)
                self._speech_start = pad_start
                self._speech_buffer = []
            self._silence_samples = 0
            self._speech_buffer.extend(chunk.tolist())
        else:
            if self._triggered:
                self._silence_samples += self.CHUNK_SIZE
                self._speech_buffer.extend(chunk.tolist())

                if self._silence_samples >= self.min_silence_samples:
                    # Silence threshold reached → emit completed segment
                    speech_arr = np.array(self._speech_buffer, dtype=np.float32)
                    # Trim trailing silence, keep a short post-roll pad
                    keep = len(speech_arr) - self._silence_samples + self.speech_pad_samples
                    keep = max(0, min(keep, len(speech_arr)))
                    speech_arr = speech_arr[:keep]

                    if len(speech_arr) >= self.min_speech_samples:
                        end_sample = self._speech_start + len(speech_arr)
                        completed.append(SpeechSegment(
                            samples=speech_arr,
                            start_sample=self._speech_start,
                            end_sample=end_sample,
                            sample_rate=self.sample_rate,
                        ))

                    self._triggered = False
                    self._silence_samples = 0
                    self._speech_buffer = []

        self._sample_offset += self.CHUNK_SIZE
        return completed

    def flush(self) -> List[SpeechSegment]:
        """
        Drain any buffered speech at end-of-stream.
        Always call after the last chunk of a file or recording session.
        """
        completed: List[SpeechSegment] = []
        if self._triggered and self._speech_buffer:
            speech_arr = np.array(self._speech_buffer, dtype=np.float32)
            if len(speech_arr) >= self.min_speech_samples:
                end_sample = self._speech_start + len(speech_arr)
                completed.append(SpeechSegment(
                    samples=speech_arr,
                    start_sample=self._speech_start,
                    end_sample=end_sample,
                    sample_rate=self.sample_rate,
                ))
        self.reset_state()
        return completed

    # ------------------------------------------------------------------
    # File mode (convenience)
    # ------------------------------------------------------------------

    def process_audio(self, audio: np.ndarray) -> List[SpeechSegment]:
        """
        Process a complete audio array and return all speech segments.

        Pads to a multiple of CHUNK_SIZE automatically.

        Args:
            audio: float32 numpy array, mono, sample_rate Hz
        """
        self.reset_state()
        audio = audio.astype(np.float32)

        # Pad to multiple of CHUNK_SIZE
        remainder = len(audio) % self.CHUNK_SIZE
        if remainder:
            audio = np.pad(audio, (0, self.CHUNK_SIZE - remainder))

        segments: List[SpeechSegment] = []
        for i in range(0, len(audio), self.CHUNK_SIZE):
            segments.extend(self.process_chunk(audio[i : i + self.CHUNK_SIZE]))
        segments.extend(self.flush())
        return segments
