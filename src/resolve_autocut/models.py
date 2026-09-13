"""
Data models for Resolve AutoCut.

This module defines the core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from fractions import Fraction
import json


class FillerType(Enum):
    """Types of filler words detected by UHM."""
    NOT_FILLER = "not_filler"
    UH = "uh"
    UM = "um"
    HMM = "hmm"
    AND = "and"
    OTHER = "other"


# Default filler types that should be considered for automatic removal
DEFAULT_REMOVABLE_FILLERS = {FillerType.UH, FillerType.UM, FillerType.HMM}


@dataclass
class Interval:
    """A time interval with start and end times.
    
    All times are in seconds (float).
    The interval is [start, end) - inclusive of start, exclusive of end.
    """
    start: float
    end: float
    
    def __post_init__(self):
        # Allow zero-duration intervals (start == end) but not reversed
        if self.start > self.end:
            raise ValueError(f"Interval start ({self.start}) > end ({self.end})")
    
    @property
    def duration(self) -> float:
        """Duration of the interval in seconds."""
        return self.end - self.start
    
    def contains(self, timestamp: float) -> bool:
        """Check if a timestamp falls within this interval."""
        return self.start <= timestamp < self.end
    
    def overlaps(self, other: 'Interval') -> bool:
        """Check if this interval overlaps with another.
        
        Two intervals [a, b) and [c, d) overlap if a < d and c < b.
        They DO overlap if they are adjacent (b == c or a == d) only if we consider
        touching intervals as overlapping. By default, [0,2) and [2,4) do NOT overlap.
        """
        return self.start < other.end and other.start < self.end
    
    def clamp(self, min_start: float, max_end: float) -> 'Interval':
        """Clamp this interval to the given bounds.
        
        If the interval is completely outside the bounds, returns a zero-duration
        interval at the clamped start position.
        """
        clamped_start = max(self.start, min_start)
        clamped_end = min(self.end, max_end)
        
        # If clamping results in start > end, create a zero-duration interval
        if clamped_start >= clamped_end:
            return Interval(start=clamped_start, end=clamped_start)
        
        return Interval(start=clamped_start, end=clamped_end)
    
    def pad(self, padding: float) -> 'Interval':
        """Add padding to both ends of the interval."""
        return Interval(
            start=self.start - padding,
            end=self.end + padding
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"start": self.start, "end": self.end}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Interval':
        """Create from dictionary."""
        return cls(start=data["start"], end=data["end"])
    
    def __repr__(self) -> str:
        return f"Interval({self.start:.4f}, {self.end:.4f})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return False
        return self.start == other.start and self.end == other.end
    
    def __lt__(self, other: 'Interval') -> bool:
        return self.start < other.start


@dataclass
class FillerDetection:
    """A single filler word detection from UHM.
    
    Attributes:
        start: Start time in seconds (absolute)
        end: End time in seconds (absolute)
        filler_type: The type of filler detected
        confidence: Confidence score (0.0 to 1.0)
        enabled: Whether this detection should be removed
    """
    start: float
    end: float
    filler_type: FillerType
    confidence: float
    enabled: bool = True
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"FillerDetection start ({self.start}) > end ({self.end})")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    @property
    def duration(self) -> float:
        """Duration of the filler in seconds."""
        return self.end - self.start
    
    @property
    def interval(self) -> Interval:
        """Convert to Interval."""
        return Interval(start=self.start, end=self.end)
    
    def pad(self, padding: float) -> 'FillerDetection':
        """Add padding to both ends."""
        return FillerDetection(
            start=self.start - padding,
            end=self.end + padding,
            filler_type=self.filler_type,
            confidence=self.confidence,
            enabled=self.enabled
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "filler_type": self.filler_type.value,
            "confidence": self.confidence,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FillerDetection':
        """Create from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            filler_type=FillerType(data["filler_type"]),
            confidence=data["confidence"],
            enabled=data.get("enabled", True)
        )
    
    def __repr__(self) -> str:
        return f"FillerDetection({self.start:.4f}-{self.end:.4f}, {self.filler_type.value}, conf={self.confidence:.3f}, enabled={self.enabled})"


@dataclass
class Word:
    """A single word from transcription with timestamp.
    
    Attributes:
        text: The word text
        start: Start time in seconds
        end: End time in seconds
        index: Position in the transcript
    """
    text: str
    start: float
    end: float
    index: int = -1
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"Word start ({self.start}) > end ({self.end})")
    
    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "index": self.index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Word':
        """Create from dictionary."""
        return cls(
            text=data["text"],
            start=data["start"],
            end=data["end"],
            index=data.get("index", -1)
        )
    
    def __repr__(self) -> str:
        return f"Word({self.text!r}, {self.start:.4f}-{self.end:.4f})"


@dataclass
class Caption:
    """A caption/subtitle segment.
    
    Attributes:
        index: Caption number (1-indexed)
        start: Start time in seconds
        end: End time in seconds
        text: The caption text
        source_words: List of word indices that contribute to this caption
    """
    index: int
    start: float
    end: float
    text: str
    source_words: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"Caption start ({self.start}) > end ({self.end})")
    
    @property
    def duration(self) -> float:
        """Duration of the caption in seconds."""
        return self.end - self.start
    
    def to_srt(self, index: Optional[int] = None) -> str:
        """Format as SRT entry."""
        idx = index if index is not None else self.index
        start_hms = self._format_time(self.start)
        end_hms = self._format_time(self.end)
        # SRT cues are separated by a blank line; without it many importers
        # parse the entire file as one cue.
        return f"{idx}\n{start_hms} --> {end_hms}\n{self.text}\n\n"
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS,mmm for SRT."""
        total_millis = max(0, int(round(seconds * 1000)))
        hours, remainder = divmod(total_millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "source_words": self.source_words
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Caption':
        """Create from dictionary."""
        return cls(
            index=data["index"],
            start=data["start"],
            end=data["end"],
            text=data["text"],
            source_words=data.get("source_words", [])
        )
    
    def __repr__(self) -> str:
        return f"Caption({self.index}, {self.start:.4f}-{self.end:.4f}, {self.text!r})"


@dataclass
class MediaInfo:
    """Information about the source media file.
    
    Attributes:
        path: Path to the media file
        duration: Total duration in seconds
        video_stream: Video stream info (if present)
        audio_stream: Audio stream info (if present)
        frame_rate: Frame rate as a Fraction (numerator, denominator)
        timebase: Timebase as a Fraction
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        width: Video width in pixels
        height: Video height in pixels
    """
    path: str
    duration: float
    frame_rate: Optional[Fraction] = None
    timebase: Optional[Fraction] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "duration": self.duration,
            "frame_rate": str(self.frame_rate) if self.frame_rate else None,
            "timebase": str(self.timebase) if self.timebase else None,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MediaInfo':
        """Create from dictionary."""
        return cls(
            path=data["path"],
            duration=data["duration"],
            frame_rate=Fraction(data["frame_rate"]) if data.get("frame_rate") else None,
            timebase=Fraction(data["timebase"]) if data.get("timebase") else None,
            sample_rate=data.get("sample_rate"),
            channels=data.get("channels"),
            width=data.get("width"),
            height=data.get("height"),
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        video_info = f"{self.width}x{self.height}@{self.frame_rate}" if self.width else "none"
        audio_info = f"{self.sample_rate}Hz/{self.channels}ch" if self.sample_rate else "none"
        return f"MediaInfo({self.path!r}, duration={self.duration:.2f}s, video={video_info}, audio={audio_info})"


@dataclass
class AnalysisResult:
    """Complete analysis result for a media file.
    
    This is the main data structure that holds all analysis information.
    """
    # Source information
    media_info: MediaInfo
    
    # Filler detection
    filler_detections: List[FillerDetection] = field(default_factory=list)
    
    # Transcription
    words: List[Word] = field(default_factory=list)
    full_transcript: str = ""
    
    # Captions
    captions: List[Caption] = field(default_factory=list)
    
    # Edit decisions
    cut_padding: float = 0.05  # 50ms default
    approved_cut_intervals: List[Interval] = field(default_factory=list)
    keep_intervals: List[Interval] = field(default_factory=list)
    
    # Settings used
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Application info
    app_version: str = "0.1.0"
    created_at: str = ""
    
    def __post_init__(self):
        from datetime import datetime
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    @property
    def total_removed_duration(self) -> float:
        """Total duration of all approved cuts."""
        return sum(interval.duration for interval in self.approved_cut_intervals)
    
    @property
    def edited_duration(self) -> float:
        """Duration of the edited timeline."""
        return self.media_info.duration - self.total_removed_duration
    
    @property
    def enabled_detections(self) -> List[FillerDetection]:
        """Get only enabled filler detections."""
        return [d for d in self.filler_detections if d.enabled]
    
    @property
    def disabled_detections(self) -> List[FillerDetection]:
        """Get only disabled filler detections."""
        return [d for d in self.filler_detections if not d.enabled]
    
    def get_cut_intervals_from_detections(self, padding: Optional[float] = None) -> List[Interval]:
        """Convert enabled filler detections to cut intervals with optional padding."""
        pad = padding if padding is not None else self.cut_padding
        intervals = []
        for det in self.enabled_detections:
            if pad > 0:
                intervals.append(det.pad(pad).interval)
            else:
                intervals.append(det.interval)
        return intervals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire analysis to dictionary."""
        return {
            "app_version": self.app_version,
            "created_at": self.created_at,
            "media_info": self.media_info.to_dict(),
            "settings": self.settings,
            "cut_padding": self.cut_padding,
            "filler_detections": [d.to_dict() for d in self.filler_detections],
            "words": [w.to_dict() for w in self.words],
            "full_transcript": self.full_transcript,
            "captions": [c.to_dict() for c in self.captions],
            "approved_cut_intervals": [i.to_dict() for i in self.approved_cut_intervals],
            "keep_intervals": [i.to_dict() for i in self.keep_intervals]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary."""
        result = cls(
            media_info=MediaInfo.from_dict(data["media_info"]),
            settings=data.get("settings", {}),
            cut_padding=data.get("cut_padding", 0.05),
            app_version=data.get("app_version", "0.1.0"),
            created_at=data.get("created_at", ""),
            full_transcript=data.get("full_transcript", "")
        )
        result.filler_detections = [
            FillerDetection.from_dict(d) for d in data.get("filler_detections", [])
        ]
        result.words = [Word.from_dict(w) for w in data.get("words", [])]
        result.captions = [Caption.from_dict(c) for c in data.get("captions", [])]
        result.approved_cut_intervals = [
            Interval.from_dict(i) for i in data.get("approved_cut_intervals", [])
        ]
        result.keep_intervals = [
            Interval.from_dict(i) for i in data.get("keep_intervals", [])
        ]
        return result
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AnalysisResult':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        return f"AnalysisResult({self.media_info.path!r}, {len(self.filler_detections)} fillers, {len(self.words)} words, {len(self.captions)} captions)"
