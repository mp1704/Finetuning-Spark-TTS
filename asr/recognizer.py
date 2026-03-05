from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

try:
    import sherpa_onnx
except ImportError as e:
    raise ImportError(
        "sherpa_onnx is not installed.\n"
        "Run: uv pip install sherpa-onnx"
    ) from e

from asr.config import ASRConfig

if TYPE_CHECKING:
    from asr.vad import SpeechSegment


class ZipformerRecognizer:
    """
    Sherpa-ONNX wrapper for the hynt/Zipformer-30M-RNNT-6000h Vietnamese ASR model.

    The model metadata shows: model_type=zipformer2, comment=non-streaming zipformer2
    Therefore OfflineRecognizer (not OnlineRecognizer) is the correct API.

    Usage: transcribe(samples) — one complete speech segment per call.
    VAD (SileroVAD) handles segmentation upstream; this class only does ASR.
    """

    def __init__(self, config: ASRConfig):
        config.validate()
        self.config = config
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(config.tokens_path),
            encoder=str(config.encoder_path),
            decoder=str(config.decoder_path),
            joiner=str(config.joiner_path),
            num_threads=config.num_threads,
            provider=config.provider,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        """
        Transcribe a complete speech segment.

        Args:
            samples: float32 numpy array, mono, 16 kHz, range [-1.0, 1.0]

        Returns:
            Transcribed Vietnamese text (may be empty for silence/noise).
        """
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.config.sample_rate, samples.astype(np.float32))
        self._recognizer.decode_streams([stream])
        return stream.result.text.strip()

    def transcribe_segments(
        self, segments: List[SpeechSegment]
    ) -> List[Tuple[SpeechSegment, str]]:
        """
        Transcribe a list of VAD-detected speech segments.

        Returns:
            List of (segment, text) tuples preserving order and timing metadata.
        """
        return [(seg, self.transcribe(seg.samples)) for seg in segments]
