"""
Media inspection and audio extraction for Resolve AutoCut.

This module uses FFmpeg/FFprobe to:
- Inspect media files
- Extract audio for analysis
- Validate media compatibility
"""

import subprocess
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from fractions import Fraction

from .models import MediaInfo


class MediaError(Exception):
    """Exception raised for media-related errors."""
    pass


class FFmpegNotFoundError(MediaError):
    """Exception raised when FFmpeg/FFprobe is not installed."""
    pass


class UnsupportedMediaError(MediaError):
    """Exception raised when media format is not supported."""
    pass


class FFmpegError(MediaError):
    """Exception raised when FFmpeg command fails."""
    pass


def check_ffmpeg() -> bool:
    """Check if FFmpeg and FFprobe are available."""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      capture_output=True, check=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], 
                      capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_ffmpeg_version() -> str:
    """Get FFmpeg version string."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith("ffmpeg version"):
                return line.strip()
        return "Unknown"
    except Exception as e:
        raise FFmpegNotFoundError(f"Cannot determine FFmpeg version: {e}")


def inspect_media(file_path: str) -> MediaInfo:
    """Inspect a media file and return its information.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        MediaInfo object with media details
        
    Raises:
        FFmpegNotFoundError: If FFmpeg/FFprobe is not available
        FFmpegError: If inspection fails
        UnsupportedMediaError: If media format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Media file not found: {file_path}")
    
    if not check_ffmpeg():
        raise FFmpegNotFoundError(
            "FFmpeg and FFprobe must be installed. "
            "Install via Homebrew: brew install ffmpeg"
        )
    
    # Use ffprobe to get media information
    try:
        # Get format and stream information
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise FFmpegError(f"FFprobe failed: {result.stderr}")
        
        info = json.loads(result.stdout)
        
    except json.JSONDecodeError as e:
        raise FFmpegError(f"Failed to parse FFprobe output: {e}")
    
    # Parse the information
    format_info = info.get("format", {})
    streams = info.get("streams", [])
    
    duration = float(format_info.get("duration", 0))
    
    # Extract video and audio stream info
    video_stream = None
    audio_stream = None
    
    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            video_stream = stream
        elif codec_type == "audio":
            audio_stream = stream
    
    # Parse frame rate
    frame_rate = None
    timebase = None
    
    if video_stream:
        # Try to get frame rate
        r_frame_rate = video_stream.get("r_frame_rate")
        avg_frame_rate = video_stream.get("avg_frame_rate")
        tb = video_stream.get("time_base")
        
        if r_frame_rate and r_frame_rate != "0/0":
            try:
                num, den = map(int, r_frame_rate.split('/'))
                frame_rate = Fraction(num, den)
            except (ValueError, ZeroDivisionError):
                pass
        
        if not frame_rate and avg_frame_rate and avg_frame_rate != "0/0":
            try:
                num, den = map(int, avg_frame_rate.split('/'))
                frame_rate = Fraction(num, den)
            except (ValueError, ZeroDivisionError):
                pass
        
        if tb and tb != "0/0":
            try:
                num, den = map(int, tb.split('/'))
                timebase = Fraction(num, den)
            except (ValueError, ZeroDivisionError):
                pass
    
    # Parse audio info
    sample_rate = None
    channels = None
    
    if audio_stream:
        sample_rate = audio_stream.get("sample_rate")
        if sample_rate:
            sample_rate = int(sample_rate)
        
        channels = audio_stream.get("channels")
        if channels:
            channels = int(channels)
    
    # Get video dimensions
    width = None
    height = None
    
    if video_stream:
        width = video_stream.get("width")
        if width:
            width = int(width)
        
        height = video_stream.get("height")
        if height:
            height = int(height)
    
    # Extract metadata
    metadata = {}
    if "tags" in format_info:
        metadata = format_info["tags"].copy()
    
    return MediaInfo(
        path=file_path,
        duration=duration,
        frame_rate=frame_rate,
        timebase=timebase,
        sample_rate=sample_rate,
        channels=channels,
        width=width,
        height=height,
        metadata=metadata
    )


def extract_audio(
    media_path: str,
    output_path: str,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    force: bool = False
) -> str:
    """Extract audio from media file as WAV.
    
    Args:
        media_path: Path to source media file
        output_path: Path to save extracted audio
        target_sample_rate: Target sample rate in Hz (default 16000)
        target_channels: Target number of channels (default 1 for mono)
        start_time: Optional start time in seconds
        end_time: Optional end time in seconds
        force: Overwrite output if it exists
        
    Returns:
        Path to the extracted audio file
        
    Raises:
        FFmpegNotFoundError: If FFmpeg is not available
        FFmpegError: If extraction fails
    """
    if not check_ffmpeg():
        raise FFmpegNotFoundError(
            "FFmpeg must be installed. Install via Homebrew: brew install ffmpeg"
        )
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Skip if output exists and not forced
    if os.path.exists(output_path) and not force:
        return output_path
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", media_path,
        "-vn",  # Disable video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(target_sample_rate),
        "-ac", str(target_channels),
        "-f", "wav",
    ]
    
    # Add time trimming if specified
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    if end_time is not None and start_time is not None:
        cmd.extend(["-to", str(end_time)])
    elif end_time is not None:
        cmd.extend(["-t", str(end_time - (start_time or 0))])
    
    cmd.append(output_path)
    
    # Run FFmpeg
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise FFmpegError(
                f"FFmpeg extraction failed:\n{result.stderr}"
            )
        
        return output_path
        
    except subprocess.TimeoutExpired:
        raise FFmpegError("FFmpeg extraction timed out")


def extract_audio_segment(
    media_path: str,
    segment_start: float,
    segment_end: float,
    output_path: str
) -> str:
    """Extract a specific time segment of audio from media.
    
    Args:
        media_path: Path to source media
        segment_start: Start time in seconds
        segment_end: End time in seconds
        output_path: Output path for the segment
        
    Returns:
        Path to extracted segment
    """
    return extract_audio(
        media_path,
        output_path,
        start_time=segment_start,
        end_time=segment_end
    )


def validate_media(file_path: str) -> tuple[bool, str]:
    """Validate if a media file can be processed.
    
    Args:
        file_path: Path to media file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        if not check_ffmpeg():
            return False, "FFmpeg/FFprobe not installed"
        
        # Try to inspect
        info = inspect_media(file_path)
        
        if info.duration <= 0:
            return False, "Invalid duration"
        
        if not info.sample_rate:
            return False, "No audio stream found"
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


class AudioExtractor:
    """Context manager for extracting audio with cleanup."""
    
    def __init__(
        self,
        media_path: str,
        target_sample_rate: int = 16000,
        target_channels: int = 1
    ):
        self.media_path = media_path
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.temp_dir = None
        self.output_path = None
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="autocut_audio_")
        self.output_path = os.path.join(self.temp_dir, "audio.wav")
        
        extract_audio(
            self.media_path,
            self.output_path,
            target_sample_rate=self.target_sample_rate,
            target_channels=self.target_channels
        )
        
        return self.output_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        return False


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    if not check_ffmpeg():
        raise FFmpegNotFoundError("FFmpeg must be installed")
    
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise FFmpegError(f"Failed to get audio duration: {result.stderr}")
        
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        return duration
        
    except Exception as e:
        raise FFmpegError(f"Cannot get audio duration: {e}")


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """Get detailed information about an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio properties
    """
    if not check_ffmpeg():
        raise FFmpegNotFoundError("FFmpeg must be installed")
    
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise FFmpegError(f"FFprobe failed: {result.stderr}")
        
        return json.loads(result.stdout)
        
    except Exception as e:
        raise FFmpegError(f"Cannot get audio info: {e}")
