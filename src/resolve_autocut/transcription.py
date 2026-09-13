"""Local MLX Whisper transcription with required real word timestamps."""

import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import logging

from .models import Word
from .media import extract_audio

logger = logging.getLogger(__name__)
DEFAULT_WHISPER_MODEL = "base"

# These are the compact MLX conversions published for the model sizes offered
# in the GUI.  Keep the GUI choices derived from this mapping rather than
# reconstructing repository names at call time.
MLX_WHISPER_MODEL_REPOSITORIES = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
}
GUI_WHISPER_MODEL_SIZES = tuple(MLX_WHISPER_MODEL_REPOSITORIES)
_HF_REPOSITORY_ID = re.compile(r"^[^/\s]+/[^/\s]+$")


class TranscriptionError(Exception): pass
class WhisperNotAvailableError(TranscriptionError): pass
class MLXNotAvailableError(TranscriptionError): pass


def resolve_whisper_model_repo(model_size: str) -> str:
    """Resolve a GUI model size or explicit Hugging Face repository ID.

    Public, application-supported sizes are deliberately mapped explicitly:
    mlx-whisper expects MLX-converted weights, whose repository names end in
    ``-mlx``.  Advanced callers may still supply a full ``owner/repository``
    ID; it is passed through unchanged so Hugging Face can report any genuine
    access error (for example for a private or gated repository).
    """
    if not isinstance(model_size, str):
        raise ValueError("Whisper model must be a model size or Hugging Face repository ID")
    if model_size in MLX_WHISPER_MODEL_REPOSITORIES:
        return MLX_WHISPER_MODEL_REPOSITORIES[model_size]
    if _HF_REPOSITORY_ID.fullmatch(model_size):
        return model_size
    choices = ", ".join(MLX_WHISPER_MODEL_REPOSITORIES)
    raise ValueError(
        f"Unknown Whisper model {model_size!r}. Choose one of: {choices}, "
        "or provide a full Hugging Face repository ID (owner/repository)."
    )


class WhisperTranscriber:
    """Thin wrapper around mlx-whisper's supported public API.

    mlx-whisper 0.4.x loads the model through ``transcribe`` using a Hugging
    Face repository, rather than exposing ``load_model``/``load_processor``.
    """
    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL):
        self.model_size = model_size
        self._model_repo = resolve_whisper_model_repo(model_size)
        self._loaded = False

    @property
    def model_repo(self) -> str:
        return self._model_repo

    @property
    def is_loaded(self) -> bool:
        # MLX Whisper owns its model cache; a completed invocation proves loading.
        return self._loaded

    def transcribe(self, audio_path: str, language: str = "en", temperature: float = 0.0,
                   word_timestamps: bool = True) -> Dict[str, Any]:
        if not word_timestamps:
            raise ValueError("Resolve AutoCut requires real word timestamps; disabling them is unsupported")
        try:
            import mlx_whisper
        except ImportError as exc:
            raise MLXNotAvailableError("mlx-whisper and MLX are required for local transcription") from exc
        try:
            result = mlx_whisper.transcribe(
                audio_path, path_or_hf_repo=self.model_repo, language=language,
                temperature=temperature, word_timestamps=True,
            )
            self._loaded = True
            if not _result_has_word_timestamps(result):
                raise TranscriptionError("MLX Whisper returned no word timestamps; refusing to fabricate them")
            return result
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"MLX Whisper transcription failed: {exc}") from exc

    def transcribe_array(self, audio_data, sample_rate: int, **kwargs) -> Dict[str, Any]:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        try:
            sf.write(path, audio_data, sample_rate)
            return self.transcribe(path, **kwargs)
        finally:
            if os.path.exists(path):
                os.unlink(path)


def _result_has_word_timestamps(result: Dict[str, Any]) -> bool:
    return bool(result.get("words")) or any(segment.get("words") for segment in result.get("segments", []))


def convert_to_words(result: Dict[str, Any]) -> List[Word]:
    """Read only timestamped words emitted by Whisper; never interpolate them."""
    raw_words = result.get("words", [])
    if not raw_words:
        raw_words = [word for segment in result.get("segments", []) for word in segment.get("words", [])]
    words = []
    for info in raw_words:
        if not {"word", "start", "end"}.issubset(info):
            raise TranscriptionError(f"Malformed MLX Whisper word result: {info!r}")
        start, end = float(info["start"]), float(info["end"])
        if end < start:
            raise TranscriptionError(f"Invalid MLX Whisper word range: {info!r}")
        words.append(Word(str(info["word"]).strip(), start, end, len(words)))
    if not words:
        raise TranscriptionError("MLX Whisper returned no timestamped words")
    return words


def get_full_transcript(result: Dict[str, Any]) -> str:
    return str(result.get("text") or " ".join(str(s.get("text", "")) for s in result.get("segments", []))).strip()


def transcribe_media(media_path: str, model_size: str = DEFAULT_WHISPER_MODEL,
                     language: str = "en", extract_audio_to: Optional[str] = None) -> List[Word]:
    owns_audio = extract_audio_to is None
    if owns_audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            audio_path = handle.name
    else:
        audio_path = extract_audio_to
    try:
        extract_audio(media_path, audio_path, force=True)
        return convert_to_words(WhisperTranscriber(model_size).transcribe(audio_path, language=language))
    finally:
        if owns_audio and os.path.exists(audio_path):
            os.unlink(audio_path)


def transcribe_with_fallback(media_path: str, model_size: str = DEFAULT_WHISPER_MODEL,
                             language: str = "en") -> Tuple[str, List[Word]]:
    """Compatibility entry point; MLX is the only supported backend in this MVP."""
    words = transcribe_media(media_path, model_size, language)
    return " ".join(word.text for word in words), words


# Backward-compatible private alias for callers from the original scaffold.
_convert_to_words = convert_to_words
