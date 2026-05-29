from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soxr

from ..interfaces import TTS

try:
    from TTS.api import TTS as CoquiTTS  # type: ignore
except Exception as e:  # pragma: no cover - early import failure
    CoquiTTS = None  # type: ignore
    _import_error = e
else:
    _import_error = None


class XTTSv2Local(TTS):
    """Local XTTS-v2 wrapper using the Coqui TTS API and local checkpoints."""

    def __init__(
        self,
        *,
        model_path: str,
        config_path: Optional[str] = None,
        voices: Optional[List[Dict[str, str]]] = None,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        sample_rate: int = 48000,
        device: str = "cuda",
        split_sentences: bool = False,
        _shared_tts: Optional[Any] = None,
        _shared_lock: Optional[Lock] = None,
    ) -> None:
        if CoquiTTS is None:  # pragma: no cover - dependency missing
            raise ImportError(f"coqui TTS is required: {_import_error}")

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.config_path = (
            str(Path(config_path).expanduser().resolve())
            if config_path
            else str(Path(self.model_path) / "config.json")
        )
        self.language = language
        self._sample_rate = int(sample_rate)
        self.device = device
        self.split_sentences = bool(split_sentences)
        self._voices = [voice.copy() for voice in (voices or [])]
        self._voice_map = {
            voice["name"]: voice["path"]
            for voice in self._voices
            if voice.get("name") and voice.get("path")
        }

        default_speaker = speaker_wav
        if not default_speaker and self._voices:
            default_speaker = self._voices[0].get("path")
        self._speaker_wav = default_speaker

        self._lock = _shared_lock or Lock()
        if _shared_tts is not None:
            self.tts = _shared_tts
        else:
            self.tts = CoquiTTS(
                model_path=self.model_path,
                config_path=self.config_path,
            )
            self.tts.to(self.device)

    def clone(self) -> "XTTSv2Local":
        return XTTSv2Local(
            model_path=self.model_path,
            config_path=self.config_path,
            voices=[voice.copy() for voice in self._voices],
            speaker_wav=self._speaker_wav,
            language=self.language,
            sample_rate=self._sample_rate,
            device=self.device,
            split_sentences=self.split_sentences,
            _shared_tts=self.tts,
            _shared_lock=self._lock,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def set_voice(self, voice_names: list[str]) -> None:
        if not voice_names:
            raise ValueError("voice_names cannot be empty for XTTSv2Local")
        voice_name = voice_names[0]
        speaker_wav = self._voice_map.get(voice_name)
        if not speaker_wav:
            raise ValueError(f"Unknown voice name: {voice_name}")
        self._speaker_wav = speaker_wav

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            raise ValueError("Text for XTTSv2Local synthesis cannot be empty.")
        if not self._speaker_wav:
            raise ValueError("speaker_wav or voices must be configured for XTTSv2Local")

        with self._lock:
            wav = self.tts.tts(
                text=text,
                speaker_wav=self._speaker_wav,
                language=self.language,
                split_sentences=self.split_sentences,
            )

        audio = np.asarray(wav, dtype=np.float32)
        model_sr = int(getattr(self.tts.synthesizer, "output_sample_rate", 24000))
        if model_sr != self.sample_rate:
            audio = soxr.resample(audio, model_sr, self.sample_rate)
        return self._float32_to_pcm_bytes(audio)

    def synthesize_stream(self, text: str, **kwargs: Any) -> Iterable[bytes]:
        yield self.synthesize(text)

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
