"""
Timeline generation for Resolve AutoCut.

This module generates FCPXML files that can be imported into DaVinci Resolve.

FCPXML is Apple's Final Cut Pro XML interchange format, which is also supported
by DaVinci Resolve for importing timelines.

Key requirements:
- Reference the original source media (no re-encoding)
- Remove approved filler regions
- Maintain video/audio synchronization
- Preserve original quality
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from xml.etree import ElementTree as ET
from xml.dom import minidom
from fractions import Fraction
import logging

from .models import MediaInfo, Interval, AnalysisResult
from .intervals import sort_intervals, merge_overlapping, invert_intervals


logger = logging.getLogger(__name__)


class TimelineError(Exception):
    """Exception raised for timeline generation errors."""
    pass


# FCPXML namespace and metadata
FCPXML_NAMESPACE = "http://apple.com/FinalCutPro/XML_Interchange/v1"
FCPXML_VERSION = "1.9"

# Default timeline settings
DEFAULT_FCPXML_FRAMERATE = "24000/1001"  # 23.976 fps (common for video)
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


def get_frame_rate_fraction(fps: float) -> Fraction:
    """Convert fps to fraction representation.
    
    Args:
        fps: Frames per second as float
        
    Returns:
        Fraction representing the frame rate
    """
    # Common frame rates
    common_rates = {
        23.976: Fraction(24000, 1001),
        24.0: Fraction(24, 1),
        25.0: Fraction(25, 1),
        29.97: Fraction(30000, 1001),
        30.0: Fraction(30, 1),
        48.0: Fraction(48, 1),
        50.0: Fraction(50, 1),
        59.94: Fraction(60000, 1001),
        60.0: Fraction(60, 1),
        16.0: Fraction(16, 1),
    }
    
    # Try exact match
    if fps in common_rates:
        return common_rates[fps]
    
    # Try to approximate
    # Round to 3 decimal places and check
    fps_rounded = round(fps, 3)
    if fps_rounded in common_rates:
        return common_rates[fps_rounded]
    
    # Fall back to simple fraction
    # Find a denominator that works
    for denom in [1, 1001, 1000, 100, 10]:
        num = round(fps * denom)
        if abs(num / denom - fps) < 0.001:
            return Fraction(int(num), denom)
    
    # Last resort: use the float as-is
    return Fraction(int(round(fps * 1000)), 1000).limit_denominator(10000)


def fraction_to_fcpxml(f: Fraction) -> str:
    """Convert a Fraction to FCPXML format (numerator/denominator).
    
    Args:
        f: Fraction to convert
        
    Returns:
        String in format "numerator/denominator"
    """
    return f"{f.numerator}/{f.denominator}"


def seconds_to_frames(seconds: float, frame_rate: Fraction) -> int:
    """Convert seconds to frame number.
    
    Args:
        seconds: Time in seconds
        frame_rate: Frame rate as Fraction
        
    Returns:
        Frame number (integer)
    """
    return int(round(seconds * float(frame_rate)))


def frames_to_seconds(frames: int, frame_rate: Fraction) -> float:
    """Convert frame number to seconds.
    
    Args:
        frames: Frame number
        frame_rate: Frame rate as Fraction
        
    Returns:
        Time in seconds
    """
    return frames / float(frame_rate)


def generate_fcpxml(
    media_info: MediaInfo,
    keep_intervals: List[Interval],
    output_path: str,
    frame_rate: Optional[Fraction] = None,
    timeline_name: str = "Resolve AutoCut Timeline"
) -> str:
    """Generate an FCPXML file from keep intervals.
    
    Args:
        media_info: MediaInfo object with source media details
        keep_intervals: List of Interval objects to keep (the edited timeline)
        output_path: Path to save the FCPXML file
        frame_rate: Optional frame rate override
        timeline_name: Name for the timeline
        
    Returns:
        Path to the generated FCPXML file
        
    Raises:
        TimelineError: If generation fails
    """
    # Determine frame rate
    if frame_rate is None:
        frame_rate = media_info.frame_rate
        if frame_rate is None:
            # Default to 24 fps
            frame_rate = Fraction(24, 1)
    
    # Ensure keep intervals are sorted and non-overlapping
    sorted_keeps = sort_intervals(keep_intervals)
    merged_keeps = merge_overlapping(sorted_keeps)
    
    # Validate that keep intervals are within media duration
    for i, keep in enumerate(merged_keeps):
        if keep.start < 0:
            raise TimelineError(f"Keep interval {i} starts before 0: {keep.start}")
        if keep.end > media_info.duration:
            raise TimelineError(
                f"Keep interval {i} extends beyond media duration ({media_info.duration}s): {keep.end}"
            )
    
    # Build the FCPXML structure
    fcpxml = ET.Element("fcpxml")
    fcpxml.set("version", FCPXML_VERSION)
    fcpxml.set("xmlns", FCPXML_NAMESPACE)
    
    # Add resources
    resources = ET.SubElement(fcpxml, "resources")
    
    # Add asset for the media file
    # Use the media file path as the reference
    media_path = media_info.path
    media_name = os.path.basename(media_path)
    
    # Create clip asset
    asset = ET.SubElement(resources, "asset")
    asset.set("id", "r1")
    asset.set("name", media_name)
    asset.set("uid", f"…/…/{media_name}")
    
    # Asset metadata
    format_ = ET.SubElement(asset, "format")
    format_.set("id", "r2")
    
    # Media file reference
    link = ET.SubElement(asset, "link")
    link.set("clipitemid", "r1")
    link.set("mediatype", "video")
    link.set("trackindex", "1")
    link.set("linkclipstillstart", "0")
    link.set("linkclipstillend", "0")
    link.set("linkcliptype", "completeclip")
    
    # Add the actual media reference
    link_media = ET.SubElement(link, "linkmedia")
    link_media.set("href", f"file://{os.path.abspath(media_path)}")
    
    # Create the library
    library = ET.SubElement(fcpxml, "library")
    
    # Add event
    event = ET.SubElement(library, "event")
    event.set("name", "Resolve AutoCut")
    
    # Add project
    project = ET.SubElement(event, "project")
    project.set("name", timeline_name)
    project.set("uid", f"…/…/{timeline_name}")
    
    # Add sequence (timeline)
    sequence = ET.SubElement(project, "sequence")
    sequence.set("duration", str(media_info.duration) + "s")
    sequence.set("format", f"{media_info.width}x{media_info.height}" if media_info.width else "1920x1080")
    sequence.set("tcStart", "00:00:00:00")
    sequence.set("tcDuration", _format_duration_fcpxml(media_info.duration, frame_rate))
    
    # Frame rate
    fr = fraction_to_fcpxml(frame_rate)
    sequence.set("rate", fr)
    
    # Add video track
    video_track = ET.SubElement(sequence, "spine")
    
    # Add clips for each keep interval
    for i, keep in enumerate(merged_keeps):
        clip = _create_fcpxml_clip(
            f"r{i+1}",
            keep,
            media_info,
            frame_rate,
            media_path
        )
        video_track.append(clip)
    
    # Add audio track(s)
    # For simplicity, we'll add a single audio track
    # Note: FCPXML can have multiple audio tracks
    
    # Write the XML
    xml_str = ET.tostring(fcpxml, encoding="unicode")
    
    # Pretty print
    xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    logger.info(f"FCPXML saved to {output_path}")
    return output_path


def _create_fcpxml_clip(
    clip_id: str,
    interval: Interval,
    media_info: MediaInfo,
    frame_rate: Fraction,
    media_path: str
) -> ET.Element:
    """Create an FCPXML clip element for a keep interval.
    
    Args:
        clip_id: Unique ID for the clip
        interval: The keep interval (source in/out)
        media_info: Media information
        frame_rate: Frame rate as Fraction
        media_path: Path to media file
        
    Returns:
        ET.Element representing the clip
    """
    clip = ET.Element("clip")
    clip.set("name", os.path.basename(media_path))
    clip.set("offset", "0s")
    clip.set("duration", f"{interval.duration}s")
    clip.set("start", f"{interval.start}s")
    
    # In and Out points (frame-based)
    in_frame = seconds_to_frames(interval.start, frame_rate)
    out_frame = seconds_to_frames(interval.end, frame_rate)
    duration_frames = out_frame - in_frame
    
    clip.set("tcIn", _format_duration_fcpxml(interval.start, frame_rate))
    clip.set("tcOut", _format_duration_fcpxml(interval.end, frame_rate))
    clip.set("tcClipIn", _format_duration_fcpxml(interval.start, frame_rate))
    clip.set("tcClipOut", _format_duration_fcpxml(interval.end, frame_rate))
    
    # Enable/disable state
    clip.set("enabled", "TRUE")
    
    # Link to the media asset
    link = ET.SubElement(clip, "link")
    link.set("linkClipRef", "r1")
    link.set("mediaType", "video")
    link.set("trackIndex", "1")
    link.set("clipIndex", "1")
    
    return clip


def _format_duration_fcpxml(seconds: float, frame_rate: Fraction) -> str:
    """Format duration for FCPXML.
    
    Args:
        seconds: Duration in seconds
        frame_rate: Frame rate as Fraction
        
    Returns:
        String in format HH:MM:SS:FF
    """
    total_frames = int(round(seconds * float(frame_rate)))
    fps = float(frame_rate)
    
    frames = int(total_frames) % int(round(fps))
    total_seconds = int(total_frames // int(round(fps)))
    
    hours = total_seconds // 3600
    remaining = total_seconds % 3600
    minutes = remaining // 60
    secs = remaining % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


def generate_fcpxml_from_analysis(
    analysis: AnalysisResult,
    output_path: str,
    timeline_name: Optional[str] = None
) -> str:
    """Generate FCPXML from a complete analysis result.
    
    Args:
        analysis: AnalysisResult object
        output_path: Path to save FCPXML
        timeline_name: Optional timeline name
        
    Returns:
        Path to generated FCPXML file
    """
    # Use keep intervals from analysis
    keep_intervals = analysis.keep_intervals
    
    if not keep_intervals:
        # If no keep intervals, use the full media
        keep_intervals = [Interval(start=0, end=analysis.media_info.duration)]
    
    name = timeline_name or f"AutoCut - {os.path.basename(analysis.media_info.path)}"
    
    return generate_fcpxml(
        analysis.media_info,
        keep_intervals,
        output_path,
        frame_rate=analysis.media_info.frame_rate,
        timeline_name=name
    )


def create_simple_fcpxml(
    media_path: str,
    cut_intervals: List[Interval],
    output_path: str,
    frame_rate: Optional[float] = None
) -> str:
    """Simple FCPXML generation with minimal parameters.
    
    Args:
        media_path: Path to source media
        cut_intervals: List of cut intervals to remove
        output_path: Path to save FCPXML
        frame_rate: Optional frame rate (fps)
        
    Returns:
        Path to generated FCPXML file
    """
    # Create media info
    from .media import inspect_media
    media_info = inspect_media(media_path)
    
    # Calculate keep intervals
    total_duration = media_info.duration
    keep_intervals = invert_intervals(cut_intervals, total_duration)
    
    # Determine frame rate
    if frame_rate:
        fr = get_frame_rate_fraction(frame_rate)
    else:
        fr = media_info.frame_rate or Fraction(24, 1)
    
    return generate_fcpxml(
        media_info,
        keep_intervals,
        output_path,
        frame_rate=fr
    )


def validate_fcpxml(fcpxml_path: str) -> Tuple[bool, List[str]]:
    """Validate an FCPXML file.
    
    Basic validation to ensure it's parseable XML.
    
    Args:
        fcpxml_path: Path to FCPXML file
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    if not os.path.exists(fcpxml_path):
        errors.append(f"File not found: {fcpxml_path}")
        return False, errors
    
    try:
        tree = ET.parse(fcpxml_path)
        root = tree.getroot()
        
        # Check for required elements
        if root.tag != "fcpxml":
            errors.append("Root element is not 'fcpxml'")
        
        # Check version
        if root.get("version") != FCPXML_VERSION:
            errors.append(f"Unexpected FCPXML version: {root.get('version')}")
        
    except ET.ParseError as e:
        errors.append(f"Invalid XML: {e}")
    except Exception as e:
        errors.append(f"Error validating FCPXML: {e}")
    
    return len(errors) == 0, errors


def calculate_timeline_duration(
    keep_intervals: List[Interval],
    media_duration: Optional[float] = None
) -> float:
    """Calculate the duration of the edited timeline.
    
    Args:
        keep_intervals: List of keep intervals
        media_duration: Optional total media duration for validation
        
    Returns:
        Total duration of the edited timeline in seconds
    """
    # Sort and merge intervals
    sorted_keeps = sort_intervals(keep_intervals)
    merged_keeps = merge_overlapping(sorted_keeps)
    
    total_duration = sum(interval.duration for interval in merged_keeps)
    
    # Validate
    if media_duration is not None:
        if abs(total_duration - media_duration) > 1.0:
            logger.warning(
                f"Calculated timeline duration ({total_duration}s) "
                f"differs significantly from media duration ({media_duration}s)"
            )
    
    return total_duration


def get_clip_info(fcpxml_path: str) -> List[Dict[str, Any]]:
    """Extract clip information from FCPXML.
    
    Args:
        fcpxml_path: Path to FCPXML file
        
    Returns:
        List of dictionaries with clip information
    """
    try:
        tree = ET.parse(fcpxml_path)
        root = tree.getroot()
        
        clips = []
        
        # Find all clip elements
        for clip in root.findall(".//clip"):
            info = {
                "name": clip.get("name", ""),
                "start": clip.get("start", "0s"),
                "duration": clip.get("duration", "0s"),
                "tcIn": clip.get("tcIn", ""),
                "tcOut": clip.get("tcOut", ""),
                "enabled": clip.get("enabled", "TRUE"),
            }
            clips.append(info)
        
        return clips
        
    except Exception as e:
        logger.error(f"Error parsing FCPXML: {e}")
        return []


class FCPXMLGenerator:
    """Class for generating FCPXML files with more control."""
    
    def __init__(self, frame_rate: Optional[Fraction] = None):
        self.frame_rate = frame_rate or Fraction(24, 1)
    
    def generate(
        self,
        media_path: str,
        keep_intervals: List[Interval],
        output_path: str,
        timeline_name: str = "Resolve AutoCut Timeline"
    ) -> str:
        """Generate FCPXML file.
        
        Args:
            media_path: Path to source media
            keep_intervals: List of intervals to keep
            output_path: Output path
            timeline_name: Name for the timeline
            
        Returns:
            Path to generated file
        """
        # Create media info
        from .media import inspect_media
        media_info = inspect_media(media_path)
        
        return generate_fcpxml(
            media_info,
            keep_intervals,
            output_path,
            frame_rate=self.frame_rate,
            timeline_name=timeline_name
        )
    
    def set_frame_rate(self, fps: float):
        """Set the frame rate.
        
        Args:
            fps: Frames per second
        """
        self.frame_rate = get_frame_rate_fraction(fps)
