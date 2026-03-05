"""
Real-time microphone streaming ASR.

Captures audio from the system microphone using sounddevice,
feeds 512-sample chunks through the VAD → ASR pipeline,
and calls a callback for each completed utterance.

Requirements:
    uv pip install sounddevice

    Linux — build PortAudio locally (no sudo needed):
        curl -LO https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
        tar xf pa_stable_v190700_20210406.tgz && cd portaudio
        ./configure --prefix=$HOME/.local && make -j$(nproc) && make install
        export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
"""
from __future__ import annotations

import queue
import sys
import threading
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as e:
    raise ImportError(
        "sounddevice is not installed.\n"
        "Run: uv pip install sounddevice\n"
        "Linux also needs PortAudio — build locally (no sudo):\n"
        "  curl -LO https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz\n"
        "  tar xf pa_stable_v190700_20210406.tgz && cd portaudio\n"
        "  ./configure --prefix=$HOME/.local && make -j$(nproc) && make install\n"
        "  export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH"
    ) from e

from asr.config import ASRConfig
from asr.pipeline import ASRPipeline, TranscriptionCallback


class MicrophoneStream:
    """
    Real-time microphone capture + ASR.

    Uses a background thread for audio capture (sounddevice) and a queue
    to decouple capture from inference, preventing dropped frames.

    Usage (callback):
        pipeline = ASRPipeline()
        mic = MicrophoneStream(pipeline)
        mic.start(on_result=lambda r: print(r.text))
        input("Press Enter to stop...")
        mic.stop()

    Usage (context manager):
        with MicrophoneStream(pipeline) as mic:
            mic.start(on_result=lambda r: print(r.text))
            input("Press Enter to stop...")
    """

    def __init__(
        self,
        pipeline: Optional[ASRPipeline] = None,
        config: Optional[ASRConfig] = None,
    ):
        if pipeline is not None:
            self.pipeline = pipeline
            self.config = pipeline.config
        else:
            self.config = config or ASRConfig()
            self.pipeline = ASRPipeline(self.config)

        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._sd_stream: Optional[sd.InputStream] = None
        self._proc_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # sounddevice callback (audio thread — must be fast, no blocking)
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status,
    ) -> None:
        if status:
            print(f"[MicrophoneStream] {status}", file=sys.stderr)
        # indata shape: (frames, channels); extract mono float32
        mono = indata[:, 0].copy()
        self._audio_queue.put_nowait(mono)

    # ------------------------------------------------------------------
    # Processing thread
    # ------------------------------------------------------------------

    def _chunk_generator(self):
        """Pull chunks from the queue; yields None sentinel terminates."""
        while True:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                if not self._running:
                    break

    def _processing_thread_fn(self, on_result: Optional[TranscriptionCallback]) -> None:
        for _ in self.pipeline.process_stream(
            self._chunk_generator(), on_result=on_result
        ):
            pass   # results delivered via on_result callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, on_result: Optional[TranscriptionCallback] = None) -> None:
        """
        Start microphone capture and ASR processing.

        Args:
            on_result: Called with each TranscriptionResult as utterances complete.
        """
        if self._running:
            raise RuntimeError("MicrophoneStream is already running.")

        self._running = True
        chunk_size = self.config.vad_chunk_size  # 512

        self._sd_stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=chunk_size,       # guarantees exactly 512 samples per callback
            device=self.config.stream_device,
            channels=1,
            dtype="float32",
            callback=self._sd_callback,
        )
        self._proc_thread = threading.Thread(
            target=self._processing_thread_fn,
            args=(on_result,),
            daemon=True,
            name="asr-proc",
        )
        self._sd_stream.start()
        self._proc_thread.start()

    def stop(self) -> None:
        """Stop capture and wait for the processing thread to finish."""
        self._running = False
        self._audio_queue.put(None)   # sentinel to unblock generator

        if self._sd_stream is not None:
            self._sd_stream.stop()
            self._sd_stream.close()
            self._sd_stream = None

        if self._proc_thread is not None:
            self._proc_thread.join(timeout=5.0)
            self._proc_thread = None

    def __enter__(self) -> "MicrophoneStream":
        return self

    def __exit__(self, *args) -> None:
        self.stop()
