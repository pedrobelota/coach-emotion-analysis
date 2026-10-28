"""Validated, per-call runtime configuration without import-time I/O."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    stt_model: str = "openai/whisper-large-v3-turbo"
    ser_model: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    ser_revision: str = "b520c9c46a719e36e1b9a91cad2cb5d0668757d8"
    backend: str = "auto"
    device: str = "auto"
    language: str = "pt"
    chunk_length_s: int = 30
    batch_size_gpu: int = 8
    batch_size_cpu: int = 1
    num_beams: int = 1
    best_of: int = 1
    use_vad: bool = True
    condition_on_previous_text: bool = False
    compute_type_cpu: str = "int8"
    compute_type_gpu: str = "float16"
    cpu_threads: int = 0
    num_workers: int = 1
    vad_min_silence_ms: int = 500

    def __post_init__(self):
        if self.backend not in {"auto", "faster-whisper", "transformers"}:
            raise ValueError("backend must be auto, faster-whisper or transformers")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        for key in (
            "stt_model",
            "ser_model",
            "ser_revision",
            "language",
            "compute_type_cpu",
            "compute_type_gpu",
        ):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        for key in (
            "chunk_length_s",
            "batch_size_gpu",
            "batch_size_cpu",
            "num_beams",
            "best_of",
            "num_workers",
            "vad_min_silence_ms",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.cpu_threads) is not int or self.cpu_threads < 0:
            raise ValueError("cpu_threads must be a nonnegative integer")
        for key in ("use_vad", "condition_on_previous_text"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")

    @classmethod
    def from_file(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)


_current: ContextVar[RuntimeConfig] = ContextVar("coach_emotion_config", default=RuntimeConfig())


def settings() -> RuntimeConfig:
    return _current.get()


@contextmanager
def configuration(config: RuntimeConfig):
    """Apply options to one call; restore them even when inference fails."""
    token = _current.set(config)
    try:
        yield config
    finally:
        _current.reset(token)
