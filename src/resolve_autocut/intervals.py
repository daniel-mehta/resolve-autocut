"""
Interval mathematics for Resolve AutoCut.

This module provides functions for manipulating time intervals with a focus on
correctness and avoiding floating-point drift.

Key concepts:
- All times are in seconds (float)
- Intervals are [start, end) - inclusive of start, exclusive of end
- Sorting, merging, clamping, and padding are provided
- Timestamp mapping handles the transformation from source to edited timeline
"""

from typing import List, Optional, Tuple
from .models import Interval


# Small epsilon for floating-point comparisons
EPSILON = 1e-10


def sort_intervals(intervals: List[Interval]) -> List[Interval]:
    """Sort intervals by start time."""
    return sorted(intervals, key=lambda x: x.start)


def merge_overlapping(intervals: List[Interval], epsilon: float = EPSILON) -> List[Interval]:
    """Merge overlapping or adjacent intervals.
    
    Args:
        intervals: List of intervals to merge
        epsilon: Tolerance for considering intervals as overlapping/adjacent
        
    Returns:
        List of merged intervals, sorted by start time
    """
    if not intervals:
        return []
    
    # Sort by start time
    sorted_intervals = sort_intervals(intervals)
    
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Check if current overlaps or is adjacent to last
        if current.start <= last.end + epsilon:
            # Merge them
            new_end = max(last.end, current.end)
            merged[-1] = Interval(start=last.start, end=new_end)
        else:
            merged.append(current)
    
    return merged


def clamp_interval(interval: Interval, min_start: float, max_end: float) -> Optional[Interval]:
    """Clamp an interval to the given bounds.
    
    Args:
        interval: The interval to clamp
        min_start: Minimum allowed start time
        max_end: Maximum allowed end time
        
    Returns:
        The clamped interval, or None if the interval is completely outside bounds
    """
    # Calculate clamped values without creating an Interval
    clamped_start = max(interval.start, min_start)
    clamped_end = min(interval.end, max_end)
    
    if clamped_start >= clamped_end - EPSILON:
        return None
    
    # Only create Interval if it's valid
    try:
        return Interval(start=clamped_start, end=clamped_end)
    except ValueError:
        return None


def clamp_intervals(intervals: List[Interval], min_start: float, max_end: float) -> List[Interval]:
    """Clamp multiple intervals to the given bounds.
    
    Args:
        intervals: List of intervals to clamp
        min_start: Minimum allowed start time
        max_end: Maximum allowed end time
        
    Returns:
        List of clamped intervals (excluding those completely outside bounds)
    """
    result = []
    for interval in intervals:
        clamped = clamp_interval(interval, min_start, max_end)
        if clamped:
            result.append(clamped)
    return result


def pad_intervals(intervals: List[Interval], padding: float, min_start: float = 0.0, max_end: float = float('inf')) -> List[Interval]:
    """Add padding to both ends of all intervals and clamp to bounds.
    
    Args:
        intervals: List of intervals to pad
        padding: Amount of padding to add to each end (seconds)
        min_start: Minimum start after padding (default 0)
        max_end: Maximum end after padding (default infinity)
        
    Returns:
        List of padded and clamped intervals
    """
    result = []
    for interval in intervals:
        padded = interval.pad(padding)
        clamped = clamp_interval(padded, min_start, max_end)
        if clamped:
            result.append(clamped)
    return result


def invert_intervals(intervals: List[Interval], total_duration: float) -> List[Interval]:
    """Compute the complement (keep intervals) of a set of intervals.
    
    Given a set of intervals and a total duration, return the intervals that
    represent the regions NOT covered by the input intervals.
    
    Args:
        intervals: List of cut intervals (sorted or unsorted)
        total_duration: Total duration of the media
        
    Returns:
        List of keep intervals (the complement)
    """
    if not intervals:
        return [Interval(start=0.0, end=total_duration)]
    
    # Sort and merge first
    sorted_intervals = sort_intervals(intervals)
    merged = merge_overlapping(sorted_intervals)
    
    keep_intervals = []
    
    # First keep interval: from 0 to first cut start
    if merged[0].start > 0:
        keep_intervals.append(Interval(start=0.0, end=merged[0].start))
    
    # Middle keep intervals: between cuts
    for i in range(len(merged) - 1):
        start = merged[i].end
        end = merged[i + 1].start
        if end > start:
            keep_intervals.append(Interval(start=start, end=end))
    
    # Last keep interval: from last cut end to total duration
    if merged[-1].end < total_duration:
        keep_intervals.append(Interval(start=merged[-1].end, end=total_duration))
    
    return keep_intervals


def calculate_removed_before(duration_map: List[Tuple[float, float]], timestamp: float) -> float:
    """Calculate total duration removed before a given timestamp.
    
    This function uses a precomputed map of (cut_start, cut_duration) pairs
    that are sorted by cut_start time.
    
    For efficiency, this uses binary search to find relevant cuts.
    
    Args:
        duration_map: List of (cut_start, cut_duration) tuples, sorted by cut_start
        timestamp: The timestamp to calculate removed duration before
        
    Returns:
        Total duration of cuts that end before or at the timestamp
    """
    if not duration_map:
        return 0.0
    
    total_removed = 0.0
    for cut_start, cut_duration in duration_map:
        cut_end = cut_start + cut_duration
        if cut_end <= timestamp + EPSILON:
            total_removed += cut_duration
        else:
            # Since the list is sorted, we can stop early
            break
    
    return total_removed


def build_duration_map(cut_intervals: List[Interval]) -> List[Tuple[float, float]]:
    """Build a duration map from cut intervals for efficient timestamp mapping.
    
    Args:
        cut_intervals: List of cut intervals
        
    Returns:
        List of (cut_start, cut_duration) tuples, sorted by cut_start
    """
    sorted_intervals = sort_intervals(cut_intervals)
    merged = merge_overlapping(sorted_intervals)
    return [(interval.start, interval.duration) for interval in merged]


def map_timestamp(source_timestamp: float, duration_map: List[Tuple[float, float]]) -> float:
    """Map a source timestamp to the edited timeline.
    
    The formula is: edited_time = source_time - total_removed_before(source_time)
    
    Args:
        source_timestamp: Timestamp in the source media
        duration_map: Precomputed duration map from build_duration_map()
        
    Returns:
        Timestamp in the edited timeline, or None if it falls in a removed section
    """
    removed_before = calculate_removed_before(duration_map, source_timestamp)
    return source_timestamp - removed_before


def map_timestamp_simple(source_timestamp: float, cut_intervals: List[Interval]) -> Optional[float]:
    """Map a source timestamp to edited timeline using simple approach.
    
    This checks if the timestamp falls within a cut interval.
    If it does, returns None (timestamp is in removed section).
    Otherwise, subtracts all cuts that start before the timestamp.
    
    Args:
        source_timestamp: Timestamp in the source media
        cut_intervals: List of cut intervals
        
    Returns:
        Timestamp in edited timeline, or None if in removed section
    """
    # Check if timestamp is in a cut interval
    for interval in cut_intervals:
        if interval.contains(source_timestamp):
            return None
    
    # Calculate total removed duration before this timestamp
    total_removed = 0.0
    for interval in cut_intervals:
        if interval.end <= source_timestamp:
            total_removed += interval.duration
    
    return source_timestamp - total_removed


def map_timestamp_with_intervals(source_timestamp: float, keep_intervals: List[Interval]) -> Optional[float]:
    """Map source timestamp to edited timeline using keep intervals.
    
    Find which keep interval contains the timestamp and compute the offset.
    
    Args:
        source_timestamp: Timestamp in source media
        keep_intervals: List of keep intervals (sorted)
        
    Returns:
        Timestamp in edited timeline, or None if in removed section
    """
    edited_start = 0.0
    sorted_keeps = merge_overlapping(sort_intervals(keep_intervals))
    for keep in sorted_keeps:
        if keep.contains(source_timestamp):
            return edited_start + source_timestamp - keep.start
        edited_start += keep.duration
    # Permit mapping the exact end of the final keep interval (for media-end
    # boundaries); interior cut starts remain excluded by half-open semantics.
    if sorted_keeps and abs(source_timestamp - sorted_keeps[-1].end) <= EPSILON:
        return edited_start
    return None


def map_caption_timestamps(caption: 'Caption', cut_intervals: List[Interval]) -> Optional['Caption']:
    """Map a caption's start and end times to the edited timeline.
    
    Args:
        caption: The caption to map (from models.py)
        cut_intervals: List of cut intervals
        
    Returns:
        A new Caption with mapped timestamps, or None if completely removed
    """
    from .models import Caption
    
    # Check if the entire caption is removed
    total_duration = None
    for interval in cut_intervals:
        if interval.contains(caption.start) and interval.contains(caption.end - EPSILON):
            return None
    
    new_start = map_timestamp_simple(caption.start, cut_intervals)
    new_end = map_timestamp_simple(caption.end, cut_intervals)
    
    if new_start is None or new_end is None:
        return None
    
    return Caption(
        index=caption.index,
        start=new_start,
        end=new_end,
        text=caption.text,
        source_words=caption.source_words
    )


def remap_intervals_to_edited(source_intervals: List[Interval], cut_intervals: List[Interval]) -> List[Interval]:
    """Remap a list of source intervals to the edited timeline.
    
    Intervals that fall completely within cut regions are excluded.
    Intervals that overlap with cuts are truncated.
    
    Args:
        source_intervals: Intervals in source timeline
        cut_intervals: Cut intervals to apply
        
    Returns:
        List of intervals in edited timeline
    """
    cuts = merge_overlapping(sort_intervals(cut_intervals))
    result: List[Interval] = []
    for src_interval in source_intervals:
        pieces = [src_interval]
        for cut in cuts:
            next_pieces = []
            for piece in pieces:
                if not piece.overlaps(cut):
                    next_pieces.append(piece)
                    continue
                if piece.start < cut.start:
                    next_pieces.append(Interval(piece.start, min(piece.end, cut.start)))
                if piece.end > cut.end:
                    next_pieces.append(Interval(max(piece.start, cut.end), piece.end))
            pieces = next_pieces
        for piece in pieces:
            removed_before_start = sum(cut.duration for cut in cuts if cut.end <= piece.start + EPSILON)
            removed_before_end = sum(cut.duration for cut in cuts if cut.end <= piece.end + EPSILON)
            mapped = Interval(piece.start - removed_before_start, piece.end - removed_before_end)
            if mapped.duration > EPSILON:
                result.append(mapped)
    return merge_overlapping(result)


def validate_intervals(intervals: List[Interval], total_duration: float) -> Tuple[bool, List[str]]:
    """Validate a list of intervals.
    
    Checks for:
    - No reversed intervals (start > end)
    - No intervals extending beyond total duration
    - No negative times
    - Overlaps (optional check)
    
    Args:
        intervals: List of intervals to validate
        total_duration: Total duration for bounds checking
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    for i, interval in enumerate(intervals):
        if interval.start >= interval.end:
            errors.append(f"Interval {i}: start ({interval.start}) >= end ({interval.end})")
        
        if interval.start < 0:
            errors.append(f"Interval {i}: negative start time ({interval.start})")
        
        if interval.end > total_duration:
            errors.append(f"Interval {i}: end ({interval.end}) > total duration ({total_duration})")
    
    # Check for overlaps
    sorted_intervals = sort_intervals(intervals)
    for i in range(len(sorted_intervals) - 1):
        if sorted_intervals[i].overlaps(sorted_intervals[i + 1]):
            errors.append(f"Intervals {i} and {i+1} overlap")
    
    return len(errors) == 0, errors


def frames_to_seconds(frame: int, frame_rate: float) -> float:
    """Convert frame number to seconds."""
    return frame / frame_rate


def seconds_to_frames(seconds: float, frame_rate: float) -> int:
    """Convert seconds to frame number (rounded to nearest integer)."""
    return round(seconds * frame_rate)


def timecode_to_seconds(timecode: str, frame_rate: float) -> float:
    """Convert timecode string (HH:MM:SS:FF) to seconds.
    
    Args:
        timecode: Timecode in format HH:MM:SS:FF or HH:MM:SS.mmm
        frame_rate: Frame rate in fps
        
    Returns:
        Time in seconds
    """
    import re
    
    # Try to parse as HH:MM:SS:FF
    match = re.match(r'(\d+):(\d+):(\d+)[.:](\d+)', timecode)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        frames_or_millis = int(match.group(4))
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        
        # Check if the last part is frames or milliseconds
        # If it's frames, convert to seconds
        if ':' in timecode and timecode.count(':') >= 2:
            # Likely HH:MM:SS:FF format
            total_seconds += frames_or_millis / frame_rate
        else:
            # Likely HH:MM:SS.mmm format
            total_seconds += frames_or_millis / 1000.0
        
        return total_seconds
    
    # Try simpler formats
    parts = timecode.split(':')
    if len(parts) == 3:
        # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        # MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    
    # Try to parse as float seconds
    try:
        return float(timecode)
    except ValueError:
        raise ValueError(f"Cannot parse timecode: {timecode}")


def seconds_to_timecode(seconds: float, frame_rate: float) -> str:
    """Convert seconds to timecode string (HH:MM:SS:FF).
    
    Args:
        seconds: Time in seconds
        frame_rate: Frame rate in fps
        
    Returns:
        Timecode string in HH:MM:SS:FF format
    """
    total_frames = int(round(seconds * frame_rate))
    frames = total_frames % int(frame_rate)
    total_seconds = total_frames // int(frame_rate)
    
    hours = total_seconds // 3600
    remaining = total_seconds % 3600
    minutes = remaining // 60
    secs = remaining % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
