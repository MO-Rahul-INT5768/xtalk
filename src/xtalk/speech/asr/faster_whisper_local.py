from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

import numpy as np

from ..interfaces import ASR
from ..utils import MockStreamRecognizer

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception as e:  # pragma: no cover - early import failure
    WhisperModel = None  # type: ignore
    _import_error = e
else:
    _import_error = None


class FasterWhisperLocal(ASR):
    """Local faster-whisper ASR backed by a CTranslate2 Whisper checkpoint."""

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 1,
        best_of: int = 1,
        temperature: float = 0.0,
        condition_on_previous_text: bool = False,
        vad_filter: bool = False,
        word_timestamps: bool = False,
        mock_window_size: int = 4,
        mock_trigger_interval_sec: float = 1.6,
        cpu_threads: int = 4,
        num_workers: int = 1,
        _shared_model: Optional[Any] = None,
        _shared_lock: Optional[Lock] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        if WhisperModel is None:  # pragma: no cover - dependency missing
            raise ImportError(f"faster-whisper is required: {_import_error}")

        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = int(beam_size)
        self.best_of = int(best_of)
        self.temperature = float(temperature)
        self.condition_on_previous_text = bool(condition_on_previous_text)
        self.vad_filter = bool(vad_filter)
        self.word_timestamps = bool(word_timestamps)
        self.cpu_threads = int(cpu_threads)
        self.num_workers = int(num_workers)
        self.extra_kwargs: Dict[str, Any] = dict(kwargs)

        self._lock = _shared_lock or Lock()
        if _shared_model is not None:
            self.model = _shared_model
        else:
            init_kwargs: Dict[str, Any] = {
                "model_size_or_path": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "cpu_threads": self.cpu_threads,
                "num_workers": self.num_workers,
            }
            if self.extra_kwargs:
                init_kwargs.update(self.extra_kwargs)
            self.model = WhisperModel(**init_kwargs)

        self._mock_recognizer = MockStreamRecognizer(
            self.async_recognize,
            window_size=mock_window_size,
            trigger_interval_sec=mock_trigger_interval_sec,
        )

    def recognize(self, audio: bytes) -> str:
        if not audio:
            return ""
        pcm = self._pcm_to_float(audio)
        try:
            with self._lock:
                segments, _ = self.model.transcribe(
                    pcm,
                    language=self.language,
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    temperature=self.temperature,
                    condition_on_previous_text=self.condition_on_previous_text,
                    vad_filter=self.vad_filter,
                    word_timestamps=self.word_timestamps,
                    without_timestamps=True,
                    task="transcribe",
                )
                text = "".join(segment.text for segment in segments).strip()
            return text
        except Exception as e:
            raise RuntimeError(f"faster-whisper recognize failed: {e}")

    def recognize_stream(self, audio: bytes, *, is_final: bool = False) -> str:
        if not audio:
            return self._mock_recognizer.recognized_text
        return self._mock_recognizer.recognize(audio, is_final=is_final)

    def stream_chunk_bytes_hint(self) -> int | None:
        return 25600

    def reset(self) -> None:
        self._mock_recognizer.reset()

    def clone(self) -> "FasterWhisperLocal":
        return FasterWhisperLocal(
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=self.vad_filter,
            word_timestamps=self.word_timestamps,
            _shared_model=self.model,
            _shared_lock=self._lock,
            **self.extra_kwargs,
        )

    @staticmethod
    def _pcm_to_float(pcm: bytes) -> np.ndarray:
        if not pcm:
            return np.zeros((0,), dtype=np.float32)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return audio / 32768.0
