"""Official ONNX integration for Desert Ant Labs' UHM model.

UHM accepts one 30 second (480,000 sample) 16 kHz mono window and returns
``probs`` shaped ``(1, 1499, 6)``: softmax probabilities every 20 ms.
Long recordings advance by the 29.98-second output coverage. Adjacent inputs
therefore share 20 ms of context while their output timelines are contiguous.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from .models import FillerDetection, FillerType

UHM_REPO = "desert-ant-labs/uhm"
UHM_FILENAME = "uhm-web-fp16.onnx"
UHM_EXPECTED_SAMPLE_RATE = 16_000
UHM_WINDOW_SIZE = 30.0
UHM_WINDOW_SAMPLES = int(UHM_WINDOW_SIZE * UHM_EXPECTED_SAMPLE_RATE)
UHM_STRIDE = 0.02
UHM_STRIDE_SAMPLES = int(UHM_STRIDE * UHM_EXPECTED_SAMPLE_RATE)
# The published model returns 1,499 frames for 30.00 seconds. Advancing by
# the output coverage (29.98 s) intentionally overlaps 20 ms of input context
# while keeping output timestamps contiguous and leaving no blind boundary.
UHM_OUTPUT_FRAMES = 1499
UHM_HOP_SAMPLES = UHM_OUTPUT_FRAMES * UHM_STRIDE_SAMPLES
UHM_CLASSES = ["not filler", "uh", "um", "hmm", "and", "other"]
UHM_CLASS_TO_FILLER = {
    "not filler": FillerType.NOT_FILLER, "uh": FillerType.UH,
    "um": FillerType.UM, "hmm": FillerType.HMM,
    "and": FillerType.AND, "other": FillerType.OTHER,
}
DEFAULT_REMOVABLE = {FillerType.UH, FillerType.UM, FillerType.HMM}
logger = logging.getLogger(__name__)


class UHMError(Exception): pass
class ModelDownloadError(UHMError): pass
class InferenceError(UHMError): pass


class UHMModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._session: Optional[ort.InferenceSession] = None
        if model_path:
            self._load_model(model_path)

    @staticmethod
    def _get_model_path() -> str:
        try:
            return hf_hub_download(repo_id=UHM_REPO, filename=UHM_FILENAME)
        except Exception as exc:
            raise ModelDownloadError(f"Failed to download UHM model from {UHM_REPO}: {exc}") from exc

    @classmethod
    def from_cache(cls) -> "UHMModel":
        return cls(cls._get_model_path())

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        try:
            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            info = self._session.get_inputs()[0]
            if info.shape != [1, UHM_WINDOW_SAMPLES] or info.type != "tensor(float)":
                raise InferenceError(f"Unexpected UHM input schema: {info.name} {info.shape} {info.type}")
        except UHMError:
            raise
        except Exception as exc:
            raise InferenceError(f"Failed to load UHM model: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def _ensure_loaded(self) -> None:
        if self._session is None:
            self._load_model(self._get_model_path())

    def get_input_info(self) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._session
        return {"name": self._session.get_inputs()[0].name,
                "shape": self._session.get_inputs()[0].shape,
                "dtype": self._session.get_inputs()[0].type,
                "outputs": [{"name": o.name, "shape": o.shape, "dtype": o.type}
                            for o in self._session.get_outputs()],
                "classes": UHM_CLASSES, "expected_sample_rate": UHM_EXPECTED_SAMPLE_RATE}

    def predict(self, audio_data: np.ndarray, sample_rate: int = UHM_EXPECTED_SAMPLE_RATE) -> np.ndarray:
        """Run one padded/exact 30 second window and return ``(1499, 6)`` probabilities."""
        self._ensure_loaded()
        assert self._session
        if sample_rate != UHM_EXPECTED_SAMPLE_RATE:
            raise ValueError(f"UHM expects {UHM_EXPECTED_SAMPLE_RATE} Hz audio, got {sample_rate}")
        samples = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        if len(samples) > UHM_WINDOW_SAMPLES:
            raise ValueError(f"UHM accepts at most {UHM_WINDOW_SIZE:g} seconds per inference window")
        if samples.size and np.max(np.abs(samples)) > 1.0:
            samples = samples / np.max(np.abs(samples))
        padded = np.pad(samples, (0, UHM_WINDOW_SAMPLES - len(samples)))
        output = self._session.run(None, {self._session.get_inputs()[0].name: padded[None, :]})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != len(UHM_CLASSES):
            raise InferenceError(f"Unexpected UHM output shape {output.shape}; expected (1, frames, 6)")
        return output[0]


class FillerDetector:
    def __init__(self, model: Optional[UHMModel] = None):
        self.model = model or UHMModel.from_cache()

    def detect_from_file(self, audio_path: str, removable_types: set = DEFAULT_REMOVABLE,
                         confidence_threshold: float = 0.5,
                         progress_callback=None) -> List[FillerDetection]:
        import soundfile as sf
        audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if np.ndim(audio) > 1:
            audio = np.mean(audio, axis=1)
        return self.detect_from_array(
            audio, rate, removable_types, confidence_threshold, progress_callback
        )

    def detect_from_array(self, audio_data: np.ndarray, sample_rate: int,
                          removable_types: set = DEFAULT_REMOVABLE,
                          confidence_threshold: float = 0.5,
                          progress_callback=None) -> List[FillerDetection]:
        audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        if sample_rate != UHM_EXPECTED_SAMPLE_RATE:
            from scipy.signal import resample_poly
            from math import gcd
            divisor = gcd(sample_rate, UHM_EXPECTED_SAMPLE_RATE)
            audio = resample_poly(audio, UHM_EXPECTED_SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
        raw: List[Tuple[float, float, int, float]] = []
        starts = range(0, len(audio), UHM_HOP_SAMPLES)
        total_chunks = max(1, (len(audio) + UHM_HOP_SAMPLES - 1) // UHM_HOP_SAMPLES)
        for chunk_index, start_sample in enumerate(starts):
            chunk = audio[start_sample:start_sample + UHM_WINDOW_SAMPLES]
            raw.extend(self._detect_chunk(chunk, start_sample / UHM_EXPECTED_SAMPLE_RATE))
            if progress_callback:
                progress_callback((chunk_index + 1) / total_chunks)
        return self._process_predictions(raw, removable_types, confidence_threshold)

    def _detect_chunk(self, chunk_audio: np.ndarray, chunk_start_time: float) -> List[Tuple[float, float, int, float]]:
        probabilities = self.model.predict(chunk_audio)
        valid_frames = min(len(probabilities), int(np.ceil(len(chunk_audio) / (UHM_STRIDE * UHM_EXPECTED_SAMPLE_RATE))))
        result = []
        for index, row in enumerate(probabilities[:valid_frames]):
            class_index = int(np.argmax(row))
            if class_index:
                result.append((chunk_start_time + index * UHM_STRIDE,
                               min(chunk_start_time + (index + 1) * UHM_STRIDE,
                                   chunk_start_time + len(chunk_audio) / UHM_EXPECTED_SAMPLE_RATE),
                               class_index, float(row[class_index])))
        return result

    def _process_predictions(self, raw: List[Tuple[float, float, int, float]], removable_types: set,
                             confidence_threshold: float) -> List[FillerDetection]:
        if not raw:
            return []
        raw.sort(key=lambda x: (x[0], x[2]))
        groups: List[List[Tuple[float, float, int, float]]] = []
        for item in raw:
            if groups and item[2] == groups[-1][-1][2] and item[0] <= groups[-1][-1][1] + 1e-7:
                groups[-1].append(item)
            else:
                groups.append([item])
        detections = []
        for group in groups:
            confidence = max(item[3] for item in group)
            if confidence < confidence_threshold:
                continue
            filler_type = UHM_CLASS_TO_FILLER[UHM_CLASSES[group[0][2]]]
            detections.append(FillerDetection(group[0][0], group[-1][1], filler_type,
                                              confidence, filler_type in removable_types))
        return detections


def chunk_audio_for_uhm(audio_data: np.ndarray, sample_rate: int,
                        chunk_size: float = UHM_WINDOW_SIZE) -> List[Tuple[float, float, np.ndarray]]:
    """Return model windows with one output-frame of intentional input overlap."""
    size = int(chunk_size * sample_rate)
    if size <= 0 or sample_rate <= 0:
        raise ValueError("sample_rate and chunk_size must be positive")
    hop = max(1, size - int(UHM_STRIDE * sample_rate))
    return [(start / sample_rate, min(start + size, len(audio_data)) / sample_rate, audio_data[start:start + size])
            for start in range(0, len(audio_data), hop)]


def detect_fillers(audio_path: str, removable_types: set = DEFAULT_REMOVABLE,
                   confidence_threshold: float = 0.5) -> List[FillerDetection]:
    return FillerDetector().detect_from_file(audio_path, removable_types, confidence_threshold)


def filter_detections(detections: List[FillerDetection], min_duration: float = 0.1,
                      max_duration: float = 5.0, min_confidence: float = 0.3) -> List[FillerDetection]:
    return [d for d in detections if min_duration <= d.duration <= max_duration and d.confidence >= min_confidence]
