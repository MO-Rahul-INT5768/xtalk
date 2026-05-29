from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Iterable, Optional

import numpy as np
import soxr
import torch
from transformers import AutoTokenizer, VitsModel

from ..interfaces import TTS


class HFVitsLocal(TTS):
    """Local Hugging Face VITS-based TTS for MMS and similar checkpoints."""

    def __init__(
        self,
        *,
        model: str,
        sample_rate: int = 48000,
        device: str = "cuda",
        _shared_model: Optional[Any] = None,
        _shared_tokenizer: Optional[Any] = None,
        _shared_lock: Optional[Lock] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.model_name = model
        self._sample_rate = int(sample_rate)
        self.device = device
        self.extra_kwargs: Dict[str, Any] = dict(kwargs)
        self._lock = _shared_lock or Lock()

        if _shared_model is not None and _shared_tokenizer is not None:
            self.model = _shared_model
            self.tokenizer = _shared_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = VitsModel.from_pretrained(self.model_name, **self.extra_kwargs)
            self.model.to(self.device)
            self.model.eval()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            raise ValueError("Text for HFVitsLocal synthesis cannot be empty.")

        inputs = self.tokenizer(text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._lock:
            with torch.no_grad():
                waveform = self.model(**inputs).waveform[0].detach().cpu().numpy()
        model_sr = int(getattr(self.model.config, "sampling_rate", 16000))
        audio = np.asarray(waveform, dtype=np.float32)
        if model_sr != self.sample_rate:
            audio = soxr.resample(audio, model_sr, self.sample_rate)
        return self._float32_to_pcm_bytes(audio)

    def synthesize_stream(self, text: str, **kwargs: Any) -> Iterable[bytes]:
        yield self.synthesize(text)

    def clone(self) -> "HFVitsLocal":
        return HFVitsLocal(
            model=self.model_name,
            sample_rate=self.sample_rate,
            device=self.device,
            _shared_model=self.model,
            _shared_tokenizer=self.tokenizer,
            _shared_lock=self._lock,
            **self.extra_kwargs,
        )

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
