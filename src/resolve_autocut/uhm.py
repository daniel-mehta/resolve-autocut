"""
UHM (Uh-and-Um Detector) integration for Resolve AutoCut.

This module provides:
- UHM model downloading and caching
- Filler detection from audio
- Chunking of long audio for UHM processing
- Conversion to FillerDetection objects

UHM model: desert-ant-labs/uhm
Model file: uhm-web-fp16.onnx
"""

import os
import tempfile
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

from huggingface_hub import hf_hub_download
import onnxruntime as ort

from .models import FillerDetection, FillerType, Interval
from .media import get_audio_duration


# UHM model configuration
UHM_REPO = "desert-ant-labs/uhm"
UHM_FILENAME = "uhm-web-fp16.onnx"
UHM_EXPECTED_SAMPLE_RATE = 16000
UHM_WINDOW_SIZE = 30.0  # UHM analyzes audio in 30-second windows
UHM_STRIDE = 0.02  # UHM produces predictions every 20ms

# UHM class labels (based on model documentation)
UHM_CLASSES = [
    "not filler",
    "uh",
    "um",
    "hmm",
    "and",
    "other"
]

# Map from class name to FillerType
UHM_CLASS_TO_FILLER = {
    "not filler": FillerType.NOT_FILLER,
    "uh": FillerType.UH,
    "um": FillerType.UM,
    "hmm": FillerType.HMM,
    "and": FillerType.AND,
    "other": FillerType.OTHER
}

# Default removable filler types
DEFAULT_REMOVABLE = {FillerType.UH, FillerType.UM, FillerType.HMM}

logger = logging.getLogger(__name__)


class UHMError(Exception):
    """Exception raised for UHM-related errors."""
    pass


class ModelDownloadError(UHMError):
    """Exception raised when model download fails."""
    pass


class InferenceError(UHMError):
    """Exception raised when inference fails."""
    pass


class UHMModel:
    """UHM filler detection model wrapper.
    
    This class manages the UHM ONNX model and provides inference capabilities.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize UHM model.
        
        Args:
            model_path: Path to the ONNX model file. If None, will download automatically.
        """
        self.model_path = model_path
        self._session = None
        self._loaded = False
        
        if model_path:
            self._load_model(model_path)
    
    @classmethod
    def from_cache(cls) -> 'UHMModel':
        """Load UHM model from Hugging Face cache.
        
        Returns:
            UHMModel instance with cached model
        """
        model_path = cls._get_model_path()
        return cls(model_path)
    
    @staticmethod
    def _get_model_path() -> str:
        """Get or download the UHM model path.
        
        Returns:
            Path to the model file
        """
        try:
            # Try to download from Hugging Face Hub
            model_path = hf_hub_download(
                repo_id=UHM_REPO,
                filename=UHM_FILENAME
            )
            return model_path
        except Exception as e:
            raise ModelDownloadError(
                f"Failed to download UHM model from {UHM_REPO}: {e}\n\n"
                f"Please ensure you have accepted the model license at:\n"
                f"https://huggingface.co/{UHM_REPO}\n\n"
                f"You may need to run: huggingface-cli login"
            )
    
    def _load_model(self, model_path: str) -> None:
        """Load the ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load ONNX model with ONNX Runtime
            self._session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']  # Use CPU (Apple Silicon will use optimized CPU)
            )
            self._loaded = True
            logger.info(f"UHM model loaded from {model_path}")
        except Exception as e:
            raise InferenceError(f"Failed to load UHM model: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self._session is not None
    
    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded, downloading if necessary."""
        if not self._loaded:
            model_path = self._get_model_path()
            self._load_model(model_path)
    
    def get_input_info(self) -> Dict[str, Any]:
        """Get model input information.
        
        Returns:
            Dictionary with input shape and sample rate info
        """
        self._ensure_loaded()
        
        # Get input shape from model
        input_info = self._session.get_inputs()[0]
        return {
            "name": input_info.name,
            "shape": input_info.shape,
            "expected_sample_rate": UHM_EXPECTED_SAMPLE_RATE
        }
    
    def predict(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Run inference on audio data.
        
        Args:
            audio_data: 1D numpy array of audio samples (float32, normalized to [-1, 1])
            sample_rate: Sample rate of audio data
            
        Returns:
            2D numpy array of predictions (num_windows, num_classes)
        """
        self._ensure_loaded()
        
        if sample_rate != UHM_EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"UHM expects {UHM_EXPECTED_SAMPLE_RATE} Hz audio, got {sample_rate} Hz"
            )
        
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Add batch dimension and channel dimension
        # UHM expects shape (batch, channels, samples) or similar
        # Need to check actual model input shape
        input_info = self._session.get_inputs()[0]
        expected_shape = input_info.shape
        
        # Reshape audio to match expected input
        # UHM-web expects (1, N) where N is the number of samples
        input_data = audio_data.reshape(1, -1).astype(np.float32)
        
        try:
            # Run inference
            outputs = self._session.run(
                None,  # Use default outputs
                {input_info.name: input_data}
            )
            
            # Get the prediction output
            # UHM returns logits for each class
            predictions = outputs[0]  # First output is the predictions
            
            return predictions
            
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}")
    
    def predict_file(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Run inference on an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (predictions, duration_in_seconds)
        """
        import soundfile as sf
        
        # Load audio file
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        
        if len(audio_data.shape) > 1:
            # Convert to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != UHM_EXPECTED_SAMPLE_RATE:
            audio_data = self._resample(audio_data, sample_rate, UHM_EXPECTED_SAMPLE_RATE)
        
        predictions = self.predict(audio_data, UHM_EXPECTED_SAMPLE_RATE)
        duration = len(audio_data) / UHM_EXPECTED_SAMPLE_RATE
        
        return predictions, duration
    
    def _resample(self, audio_data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio data.
        
        Simple implementation using scipy.
        """
        from scipy import signal
        
        old_length = len(audio_data)
        if old_length == 0:
            return audio_data
        
        new_length = int(old_length * to_rate / from_rate)
        
        if new_length == 0:
            return np.array([], dtype=np.float32)
        
        resampled = signal.resample(audio_data, new_length)
        return resampled.astype(np.float32)


class FillerDetector:
    """High-level filler detection from audio files.
    
    Handles:
    - Chunking audio for UHM (30-second windows)
    - Running UHM inference on each chunk
    - Processing predictions into FillerDetection objects
    - Preserving absolute timestamps across chunks
    """
    
    def __init__(self, model: Optional[UHMModel] = None):
        """Initialize filler detector.
        
        Args:
            model: UHMModel instance. If None, will create one.
        """
        self.model = model or UHMModel.from_cache()
        self._chunk_size = UHM_WINDOW_SIZE  # 30 seconds
        self._stride = UHM_STRIDE  # 20ms = 0.02s
    
    def detect_from_file(
        self,
        audio_path: str,
        removable_types: set = DEFAULT_REMOVABLE,
        confidence_threshold: float = 0.5
    ) -> List[FillerDetection]:
        """Detect fillers in an audio file.
        
        Args:
            audio_path: Path to audio file
            removable_types: Set of FillerType to mark as enabled by default
            confidence_threshold: Minimum confidence for a detection
            
        Returns:
            List of FillerDetection objects with absolute timestamps
        """
        import soundfile as sf
        
        # Load audio
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sample_rate != UHM_EXPECTED_SAMPLE_RATE:
            audio_data = self._resample(audio_data, sample_rate, UHM_EXPECTED_SAMPLE_RATE)
        
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        
        # Process in chunks
        all_detections = []
        chunk_size_samples = int(self._chunk_size * sample_rate)
        stride_samples = int(self._stride * sample_rate)
        
        for chunk_start_sample in range(0, len(audio_data), stride_samples):
            chunk_end_sample = min(
                chunk_start_sample + chunk_size_samples,
                len(audio_data)
            )
            
            chunk_audio = audio_data[chunk_start_sample:chunk_end_sample]
            chunk_start_time = chunk_start_sample / sample_rate
            
            # Process chunk
            chunk_detections = self._detect_chunk(
                chunk_audio, chunk_start_time, sample_rate
            )
            all_detections.extend(chunk_detections)
        
        # Process predictions into detections
        detections = self._process_predictions(
            all_detections, removable_types, confidence_threshold
        )
        
        return detections
    
    def detect_from_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        removable_types: set = DEFAULT_REMOVABLE,
        confidence_threshold: float = 0.5
    ) -> List[FillerDetection]:
        """Detect fillers from audio array.
        
        Args:
            audio_data: 1D numpy array of audio samples
            sample_rate: Sample rate in Hz
            removable_types: Set of FillerType to mark as enabled
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of FillerDetection objects
        """
        # Resample if needed
        if sample_rate != UHM_EXPECTED_SAMPLE_RATE:
            audio_data = self._resample(audio_data, sample_rate, UHM_EXPECTED_SAMPLE_RATE)
        
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        
        # Process in chunks
        all_detections = []
        chunk_size_samples = int(self._chunk_size * sample_rate)
        stride_samples = int(self._stride * sample_rate)
        
        for chunk_start_sample in range(0, len(audio_data), stride_samples):
            chunk_end_sample = min(
                chunk_start_sample + chunk_size_samples,
                len(audio_data)
            )
            
            chunk_audio = audio_data[chunk_start_sample:chunk_end_sample]
            chunk_start_time = chunk_start_sample / sample_rate
            
            chunk_detections = self._detect_chunk(
                chunk_audio, chunk_start_time, sample_rate
            )
            all_detections.extend(chunk_detections)
        
        # Process into detections
        detections = self._process_predictions(
            all_detections, removable_types, confidence_threshold
        )
        
        return detections
    
    def _detect_chunk(
        self,
        chunk_audio: np.ndarray,
        chunk_start_time: float,
        sample_rate: int
    ) -> List[Tuple[float, float, int]]:
        """Detect fillers in a single chunk.
        
        Args:
            chunk_audio: Audio data for this chunk
            chunk_start_time: Absolute start time of this chunk
            sample_rate: Sample rate
            
        Returns:
            List of (start_time, end_time, class_index) tuples
        """
        # Run inference
        predictions = self.model.predict(chunk_audio, sample_rate)
        
        # Get num_samples and num_windows from predictions shape
        num_samples = len(chunk_audio)
        num_windows = predictions.shape[0] if predictions.size > 0 else 0
        
        detections = []
        
        # UHM produces predictions every 20ms
        # Each prediction is for a window of some duration
        # We need to map predictions to time intervals
        
        # Based on UHM documentation, predictions are every 20ms
        window_stride = 0.02  # 20ms
        
        # The output shape might be (num_windows, num_classes)
        if len(predictions.shape) == 2:
            num_windows, num_classes = predictions.shape
            
            # Get predicted class for each window
            pred_classes = np.argmax(predictions, axis=1)
            pred_confidences = np.max(predictions, axis=1)
            
            # Each prediction is for a 20ms window
            # The window center is at start + window_idx * stride
            for window_idx in range(num_windows):
                window_center = chunk_start_time + window_idx * window_stride
                window_start = window_center - window_stride / 2
                window_end = window_center + window_stride / 2
                
                class_idx = int(pred_classes[window_idx])
                confidence = float(pred_confidences[window_idx])
                
                # Only consider filler classes
                if class_idx > 0:  # 0 is "not filler"
                    detections.append((
                        window_start,
                        window_end,
                        class_idx,
                        confidence
                    ))
        
        return detections
    
    def _process_predictions(
        self,
        raw_detections: List[Tuple[float, float, int, float]],
        removable_types: set,
        confidence_threshold: float
    ) -> List[FillerDetection]:
        """Process raw predictions into FillerDetection objects.
        
        Args:
            raw_detections: List of (start, end, class_idx, confidence)
            removable_types: Set of FillerType to enable by default
            confidence_threshold: Minimum confidence
            
        Returns:
            List of FillerDetection objects
        """
        # Group consecutive predictions of the same type
        # This merges adjacent 20ms windows into longer filler regions
        
        if not raw_detections:
            return []
        
        # Sort by start time
        raw_detections.sort(key=lambda x: x[0])
        
        # Group consecutive same-class detections
        grouped = []
        current_group = [raw_detections[0]]
        
        for i in range(1, len(raw_detections)):
            prev = current_group[-1]
            curr = raw_detections[i]
            
            # Check if consecutive and same class
            if (curr[0] - prev[1] < 0.01 and  # Within 10ms gap
                curr[2] == prev[2]):  # Same class
                current_group.append(curr)
            else:
                grouped.append(current_group)
                current_group = [curr]
        
        grouped.append(current_group)
        
        # Convert groups to FillerDetection objects
        detections = []
        for group in grouped:
            start = group[0][0]
            end = group[-1][1]
            class_idx = group[0][2]
            confidence = max(d[3] for d in group)
            
            # Map class index to FillerType
            class_name = UHM_CLASSES[class_idx]
            filler_type = UHM_CLASS_TO_FILLER.get(class_name, FillerType.OTHER)
            
            # Determine if enabled by default
            enabled = filler_type in removable_types
            
            if confidence >= confidence_threshold:
                detection = FillerDetection(
                    start=start,
                    end=end,
                    filler_type=filler_type,
                    confidence=confidence,
                    enabled=enabled
                )
                detections.append(detection)
        
        return detections
    
    def _resample(self, audio_data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio data."""
        from scipy import signal
        
        old_length = len(audio_data)
        if old_length == 0:
            return audio_data
        
        new_length = int(old_length * to_rate / from_rate)
        
        if new_length == 0:
            return np.array([], dtype=np.float32)
        
        resampled = signal.resample(audio_data, new_length)
        return resampled.astype(np.float32)


def chunk_audio_for_uhm(
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_size: float = UHM_WINDOW_SIZE
) -> List[Tuple[float, float, np.ndarray]]:
    """Chunk audio for UHM processing.
    
    Divides audio into overlapping 30-second windows with 20ms stride.
    
    Args:
        audio_data: 1D numpy array of audio samples
        sample_rate: Sample rate in Hz
        chunk_size: Size of each chunk in seconds
        
    Returns:
        List of (start_time, end_time, chunk_audio) tuples
    """
    chunk_size_samples = int(chunk_size * sample_rate)
    stride_samples = int(UHM_STRIDE * sample_rate)
    
    chunks = []
    
    for chunk_start_sample in range(0, len(audio_data), stride_samples):
        chunk_end_sample = min(
            chunk_start_sample + chunk_size_samples,
            len(audio_data)
        )
        
        chunk_audio = audio_data[chunk_start_sample:chunk_end_sample]
        chunk_start_time = chunk_start_sample / sample_rate
        chunk_end_time = chunk_end_sample / sample_rate
        
        if len(chunk_audio) > 0:
            chunks.append((
                chunk_start_time,
                chunk_end_time,
                chunk_audio
            ))
    
    return chunks


def detect_fillers(
    audio_path: str,
    removable_types: set = DEFAULT_REMOVABLE,
    confidence_threshold: float = 0.5
) -> List[FillerDetection]:
    """Convenience function to detect fillers in an audio file.
    
    Args:
        audio_path: Path to audio file
        removable_types: Set of FillerType to mark as enabled
        confidence_threshold: Minimum confidence
        
    Returns:
        List of FillerDetection objects
    """
    detector = FillerDetector()
    return detector.detect_from_file(
        audio_path,
        removable_types=removable_types,
        confidence_threshold=confidence_threshold
    )


def filter_detections(
    detections: List[FillerDetection],
    min_duration: float = 0.1,
    max_duration: float = 5.0,
    min_confidence: float = 0.3
) -> List[FillerDetection]:
    """Filter filler detections by various criteria.
    
    Args:
        detections: List of FillerDetection objects
        min_duration: Minimum detection duration
        max_duration: Maximum detection duration
        min_confidence: Minimum confidence
        
    Returns:
        Filtered list of FillerDetection objects
    """
    return [
        d for d in detections
        if (d.duration >= min_duration and
            d.duration <= max_duration and
            d.confidence >= min_confidence)
    ]
