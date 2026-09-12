"""
Transcription module for Resolve AutoCut.

This module provides local speech-to-text using MLX Whisper on Apple Silicon.
It extracts word-level timestamps for accurate caption generation.
"""

import os
import tempfile
import json
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

from .models import Word, MediaInfo
from .media import extract_audio, get_audio_duration, check_ffmpeg


logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Exception raised for transcription errors."""
    pass


class WhisperNotAvailableError(TranscriptionError):
    """Exception raised when MLX Whisper is not available."""
    pass


class MLXNotAvailableError(TranscriptionError):
    """Exception raised when MLX is not available."""
    pass


# Transcription configuration
DEFAULT_WHISPER_MODEL = "base"
# Options: "tiny", "base", "small", "medium", "large"
# For Apple Silicon, "base" or "small" are good starting points

# Word timestamp extraction settings
WORD_TIMESTAMP_PRECISION = "word"  # Get word-level timestamps


class WhisperTranscriber:
    """MLX Whisper transcription wrapper.
    
    This class manages the Whisper model and provides transcription capabilities
    with word-level timestamps.
    """
    
    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL):
        """Initialize Whisper transcriber.
        
        Args:
            model_size: Size of Whisper model ("tiny", "base", "small", etc.)
        """
        self.model_size = model_size
        self._model = None
        self._processor = None
        self._loaded = False
    
    def _check_mlx(self) -> bool:
        """Check if MLX and MLX Whisper are available."""
        try:
            import mlx_whisper
            import mlx.core
            return True
        except ImportError:
            return False
    
    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if not self._loaded:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the MLX Whisper model."""
        if not self._check_mlx():
            raise MLXNotAvailableError(
                "MLX and mlx-whisper must be installed.\n\n"
                "Install via uv: uv pip install mlx mlx-whisper\n"
                "Or via pip: pip install mlx mlx-whisper"
            )
        
        try:
            import mlx_whisper
            
            # Load model
            self._model = mlx_whisper.load_model(self.model_size)
            self._processor = mlx_whisper.load_processor(self.model_size)
            self._loaded = True
            logger.info(f"MLX Whisper model '{self.model_size}' loaded")
            
        except Exception as e:
            raise WhisperNotAvailableError(
                f"Failed to load MLX Whisper model '{self.model_size}': {e}"
            )
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self._model is not None
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        temperature: float = 0.0,
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """Transcribe audio file to text with word timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default "en" for English)
            temperature: Sampling temperature (0.0 for greedy)
            word_timestamps: Whether to extract word-level timestamps
            
        Returns:
            Dictionary with:
                - "text": Full transcript
                - "words": List of word dictionaries with timestamps
                - "segments": List of segment dictionaries
        """
        self._ensure_loaded()
        
        try:
            import mlx_whisper
            
            # Transcribe
            result = mlx_whisper.transcribe(
                self._model,
                self._processor,
                audio_path,
                language=language,
                temperature=temperature,
                return_word_timestamps=word_timestamps
            )
            
            return result
            
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")
    
    def transcribe_array(
        self,
        audio_data: 'np.ndarray',
        sample_rate: int,
        language: str = "en",
        temperature: float = 0.0,
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """Transcribe audio array to text with word timestamps.
        
        Args:
            audio_data: 1D numpy array of audio samples (float32)
            sample_rate: Sample rate in Hz
            language: Language code
            temperature: Sampling temperature
            word_timestamps: Whether to extract word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        self._ensure_loaded()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Save as WAV
            import soundfile as sf
            sf.write(f.name, audio_data, sample_rate, format='WAV')
            temp_path = f.name
        
        try:
            result = self.transcribe(
                temp_path,
                language=language,
                temperature=temperature,
                word_timestamps=word_timestamps
            )
            return result
        finally:
            os.unlink(temp_path)


def transcribe_media(
    media_path: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: str = "en",
    extract_audio_to: Optional[str] = None
) -> List[Word]:
    """Transcribe media file and return word-level results.
    
    Args:
        media_path: Path to media file (video or audio)
        model_size: Whisper model size
        language: Language code
        extract_audio_to: Optional path to extract audio (for debugging)
        
    Returns:
        List of Word objects with timestamps
    """
    # Extract audio if needed
    if extract_audio_to:
        audio_path = extract_audio_to
        extract_audio(media_path, audio_path)
    else:
        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        
        try:
            extract_audio(media_path, audio_path)
            
            # Transcribe
            transcriber = WhisperTranscriber(model_size)
            result = transcriber.transcribe(audio_path, language=language)
            
            # Convert to Word objects
            words = _convert_to_words(result)
            
            return words
            
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    # If we have an audio path already
    transcriber = WhisperTranscriber(model_size)
    result = transcriber.transcribe(audio_path, language=language)
    words = _convert_to_words(result)
    return words


def _convert_to_words(result: Dict[str, Any]) -> List[Word]:
    """Convert MLX Whisper result to Word objects.
    
    Args:
        result: Dictionary from MLX Whisper transcription
        
    Returns:
        List of Word objects
    """
    words = []
    
    # Check if we have word-level timestamps
    if "words" in result and result["words"]:
        for i, word_info in enumerate(result["words"]):
            word = Word(
                text=word_info.get("word", ""),
                start=word_info.get("start", 0.0),
                end=word_info.get("end", 0.0),
                index=i
            )
            words.append(word)
    elif "segments" in result:
        # Fall back to segment-level if no word-level
        for i, segment in enumerate(result["segments"]):
            text = segment.get("text", "")
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            
            # Split segment into words
            segment_words = text.split()
            if segment_words:
                # If we only have segment timestamps, distribute them
                duration = end - start
                for j, word_text in enumerate(segment_words):
                    word_start = start + (j / len(segment_words)) * duration
                    word_end = start + ((j + 1) / len(segment_words)) * duration
                    
                    word = Word(
                        text=word_text,
                        start=word_start,
                        end=word_end,
                        index=len(words)
                    )
                    words.append(word)
    
    return words


def get_full_transcript(result: Dict[str, Any]) -> str:
    """Get the full transcript text from Whisper result.
    
    Args:
        result: Dictionary from MLX Whisper transcription
        
    Returns:
        Full transcript as a single string
    """
    if "text" in result:
        return result["text"]
    elif "segments" in result:
        return " ".join(seg.get("text", "") for seg in result["segments"])
    return ""


def transcribe_with_fallback(
    media_path: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: str = "en"
) -> Tuple[str, List[Word]]:
    """Transcribe with fallback options.
    
    Tries MLX Whisper first, falls back to other methods if available.
    
    Args:
        media_path: Path to media file
        model_size: Whisper model size
        language: Language code
        
    Returns:
        Tuple of (full_transcript, list_of_words)
    """
    # First, try MLX Whisper
    try:
        words = transcribe_media(media_path, model_size, language)
        full_text = " ".join(w.text for w in words)
        return full_text, words
    except Exception as e:
        logger.warning(f"MLX Whisper failed: {e}")
    
    # Fallback: Try to use whisper via pip (CPU version)
    try:
        import whisper
        
        model = whisper.load_model(model_size)
        result = model.transcribe(media_path, language=language, word_timestamps=True)
        
        words = []
        for i, segment in enumerate(result.get("segments", [])):
            for word_info in segment.get("words", []):
                word = Word(
                    text=word_info.get("word", ""),
                    start=word_info.get("start", 0.0),
                    end=word_info.get("end", 0.0),
                    index=len(words)
                )
                words.append(word)
        
        full_text = " ".join(w.text for w in words)
        return full_text, words
        
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Pip whisper failed: {e}")
    
    # Final fallback: return empty
    logger.error("All transcription methods failed")
    return "", []


class TranscriptionBackend:
    """Abstract base class for transcription backends."""
    
    def transcribe(self, audio_path: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Dictionary with transcription results
        """
        raise NotImplementedError
    
    def get_words(self, result: Dict[str, Any]) -> List[Word]:
        """Convert result to Word objects.
        
        Args:
            result: Transcription result dictionary
            
        Returns:
            List of Word objects
        """
        raise NotImplementedError


class MLXWhisperBackend(TranscriptionBackend):
    """MLX Whisper transcription backend."""
    
    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL):
        self.transcriber = WhisperTranscriber(model_size)
    
    def transcribe(self, audio_path: str, language: str = "en") -> Dict[str, Any]:
        return self.transcriber.transcribe(audio_path, language=language)
    
    def get_words(self, result: Dict[str, Any]) -> List[Word]:
        return _convert_to_words(result)


def create_transcriber(backend: str = "mlx", model_size: str = DEFAULT_WHISPER_MODEL) -> TranscriptionBackend:
    """Factory function to create a transcriber.
    
    Args:
        backend: Backend type ("mlx", "whisper-cpu")
        model_size: Model size
        
    Returns:
        TranscriptionBackend instance
    """
    if backend == "mlx":
        return MLXWhisperBackend(model_size)
    elif backend == "whisper-cpu":
        try:
            import whisper
            return WhisperCPUBackend(model_size)
        except ImportError:
            raise ValueError("whisper package not installed for CPU backend")
    else:
        raise ValueError(f"Unknown backend: {backend}")


class WhisperCPUBackend(TranscriptionBackend):
    """CPU-based Whisper backend (via pip whisper)."""
    
    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL):
        self.model_size = model_size
        self._model = None
    
    def _ensure_loaded(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
    
    def transcribe(self, audio_path: str, language: str = "en") -> Dict[str, Any]:
        self._ensure_loaded()
        
        result = self._model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True
        )
        
        # Convert to standard format
        return {
            "text": result.get("text", ""),
            "words": [],
            "segments": result.get("segments", [])
        }
    
    def get_words(self, result: Dict[str, Any]) -> List[Word]:
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word = Word(
                    text=word_info.get("word", ""),
                    start=word_info.get("start", 0.0),
                    end=word_info.get("end", 0.0),
                    index=len(words)
                )
                words.append(word)
        return words
