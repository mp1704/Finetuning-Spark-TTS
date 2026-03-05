from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ASRConfig:
    """Configuration for the VAD + Zipformer RNNT ASR pipeline."""

    # ---- Model location ------------------------------------------------
    model_dir: Path = field(default_factory=lambda: Path("asr/models/zipformer-30m"))
    # int8-quantised ONNX: encoder 27.7 MB vs 92.2 MB full-precision
    use_int8: bool = True

    # ---- Sherpa-ONNX recogniser ----------------------------------------
    num_threads: int = 4
    provider: str = "cpu"           # "cpu" | "cuda" | "coreml"
    sample_rate: int = 16000

    # ---- Silero VAD ----------------------------------------------------
    # chunk_size MUST be 512 samples at 16 kHz — hard constraint from Silero VAD
    vad_chunk_size: int = 512
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 100

    # ---- Microphone streaming ------------------------------------------
    stream_device: int | None = None    # None = system default mic

    # ---- Computed paths ------------------------------------------------
    @property
    def _suffix(self) -> str:
        return ".int8.onnx" if self.use_int8 else ".onnx"

    @property
    def encoder_path(self) -> Path:
        return self.model_dir / f"encoder-epoch-20-avg-10{self._suffix}"

    @property
    def decoder_path(self) -> Path:
        return self.model_dir / f"decoder-epoch-20-avg-10{self._suffix}"

    @property
    def joiner_path(self) -> Path:
        return self.model_dir / f"joiner-epoch-20-avg-10{self._suffix}"

    @property
    def bpe_model_path(self) -> Path:
        return self.model_dir / "bpe.model"

    @property
    def tokens_path(self) -> Path:
        # Generated from bpe.model by download_model.py
        return self.model_dir / "tokens.txt"

    def validate(self) -> None:
        """Raise FileNotFoundError if required model files are missing."""
        for name, path in [
            ("encoder", self.encoder_path),
            ("decoder", self.decoder_path),
            ("joiner", self.joiner_path),
            ("tokens", self.tokens_path),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {name}: {path}\n"
                    f"Download the model first:\n"
                    f"  uv run python -m asr.download_model"
                )
