"""
Caption generation and timestamp remapping for Resolve AutoCut.

This module handles:
- Generating captions from word-level transcripts
- Remapping caption timestamps after cuts are applied
- Formatting captions as SRT
"""

from typing import List, Optional, Dict, Any
from .models import Word, Caption, Interval
from .intervals import (
    sort_intervals, merge_overlapping, invert_intervals,
    map_timestamp_simple, build_duration_map, calculate_removed_before
)


# Target caption length in characters (approximate)
TARGET_CAPTION_LENGTH = 40
# Maximum caption length
MAX_CAPTION_LENGTH = 60
# Maximum caption duration in seconds
MAX_CAPTION_DURATION = 5.0
# Minimum caption duration in seconds
MIN_CAPTION_DURATION = 0.5


def generate_captions_from_words(
    words: List[Word],
    max_length: int = TARGET_CAPTION_LENGTH,
    max_duration: float = MAX_CAPTION_DURATION,
    min_duration: float = MIN_CAPTION_DURATION
) -> List[Caption]:
    """Generate captions from word-level transcript.
    
    This creates phrase-level caption segments that are:
    - Readable (group words into phrases)
    - Not too long (character limit)
    - Not too short (duration limit)
    - Properly punctuated
    
    Args:
        words: List of Word objects with timestamps
        max_length: Maximum characters per caption
        max_duration: Maximum duration per caption in seconds
        min_duration: Minimum duration per caption in seconds
        
    Returns:
        List of Caption objects
    """
    if not words:
        return []
    
    captions = []
    current_caption_words = []
    current_caption_start = words[0].start
    current_text = ""
    
    for word in words:
        # Try adding this word to current caption
        new_text = current_text + (" " if current_text else "") + word.text
        new_duration = word.end - current_caption_start
        
        # Check if adding this word would exceed limits
        if (len(new_text) > max_length or 
            new_duration > max_duration or
            (current_caption_words and new_duration > max_duration * 0.5 and 
             word.text in [".", "?", "!", ",", ";", ":"])):
            # Finalize current caption
            if current_caption_words:
                caption = _finalize_caption(
                    current_caption_words, current_caption_start, words
                )
                if caption.duration >= min_duration:
                    captions.append(caption)
            
            # Start new caption
            current_caption_words = [word.index]
            current_caption_start = word.start
            current_text = word.text
        else:
            # Add word to current caption
            current_caption_words.append(word.index)
            current_text = new_text
    
    # Finalize last caption
    if current_caption_words:
        caption = _finalize_caption(current_caption_words, current_caption_start, words)
        if caption.duration >= min_duration:
            captions.append(caption)
    
    # Re-index captions
    for i, caption in enumerate(captions):
        caption.index = i + 1
    
    return captions


def _finalize_caption(word_indices: List[int], start: float, all_words: List[Word]) -> Caption:
    """Create a Caption from a list of word indices."""
    # Get the actual words
    caption_words = [all_words[i] for i in word_indices]
    
    # Build text
    text = " ".join(w.text for w in caption_words)
    
    # Get end time
    end = max(w.end for w in caption_words)
    
    # Clean up text
    text = _clean_caption_text(text)
    
    return Caption(
        index=0,  # Will be set later
        start=start,
        end=end,
        text=text,
        source_words=word_indices
    )


def _clean_caption_text(text: str) -> str:
    """Clean up caption text for readability."""
    import re
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:] if text else text
    
    # Add period if missing at end (for sentences)
    # This is a simple heuristic - better transcription should include punctuation
    if text and text[-1] not in [".", "?", "!"]:
        # Check if it looks like a complete thought
        words = text.split()
        if len(words) > 1:
            # Add a period if the last word doesn't end with punctuation
            if not any(c in words[-1] for c in [".", "?", "!", ",", ";", ":"]):
                text = text + "."
    
    return text.strip()


def remap_captions(captions: List[Caption], cut_intervals: List[Interval], words: Optional[List[Word]] = None) -> List[Caption]:
    """Remap caption timestamps to edited timeline.
    
    This shifts all caption timestamps backward by the cumulative duration of
    cuts that occur before each caption.
    
    Captions that are completely within removed sections are excluded.
    Captions that overlap with removed sections may be truncated or excluded.
    
    Args:
        captions: List of Caption objects in source timeline
        cut_intervals: List of Interval objects representing cuts
        
    Returns:
        List of Caption objects in edited timeline
    """
    if not captions:
        return []
    
    cuts = merge_overlapping(sort_intervals(cut_intervals))
    # With source words, remove words whose audio was removed and split a
    # caption at a cut.  This avoids displaying a spoken filler after it has
    # been ripple-deleted.
    if words is not None:
        by_index = {word.index: word for word in words}
        result = []
        for caption in captions:
            run = []
            for index in caption.source_words:
                word = by_index.get(index)
                kept = word is not None and not any(word.start < cut.end and word.end > cut.start for cut in cuts)
                if kept:
                    run.append(word)
                elif run:
                    result.append(_caption_from_kept_words(caption.index, run, cuts))
                    run = []
            if run:
                result.append(_caption_from_kept_words(caption.index, run, cuts))
        return [caption for caption in result if caption.end > caption.start]

    # Legacy callers without words cannot safely rewrite caption text.
    duration_map = build_duration_map(cut_intervals)
    
    result = []
    for caption in captions:
        new_caption = _remap_single_caption(caption, cut_intervals, duration_map)
        if new_caption:
            result.append(new_caption)
    
    return result


def _caption_from_kept_words(index: int, words: List[Word], cuts: List[Interval]) -> Caption:
    start = map_timestamp_simple(words[0].start, cuts)
    end = map_timestamp_simple(words[-1].end, cuts)
    # Half-open cuts include their start but a kept word may legitimately end
    # exactly there.  Map that boundary to the beginning of the ripple gap.
    if end is None:
        for cut in cuts:
            if abs(words[-1].end - cut.start) < 1e-9:
                end = cut.start - sum(previous.duration for previous in cuts if previous.end <= cut.start)
                break
    assert start is not None and end is not None
    return Caption(index=index, start=start, end=end,
                   text=_clean_caption_text(" ".join(word.text for word in words)),
                   source_words=[word.index for word in words])


def _remap_single_caption(
    caption: Caption,
    cut_intervals: List[Interval],
    duration_map: List[tuple]
) -> Optional[Caption]:
    """Remap a single caption to edited timeline."""
    # Check if caption is completely within a cut interval
    for cut in cut_intervals:
        if cut.contains(caption.start) and cut.contains(caption.end - 1e-10):
            return None
    
    # Map start and end
    new_start = map_timestamp_simple(caption.start, cut_intervals)
    new_end = map_timestamp_simple(caption.end, cut_intervals)
    
    if new_start is None or new_end is None:
        # Caption overlaps with a cut - need to handle carefully
        # For now, try to salvage part of it
        return _remap_partial_caption(caption, cut_intervals)
    
    # Check if the caption would be too short after mapping
    if new_end - new_start < 0.1:  # Less than 100ms
        return None
    
    return Caption(
        index=caption.index,
        start=new_start,
        end=new_end,
        text=caption.text,
        source_words=caption.source_words
    )


def _remap_partial_caption(caption: Caption, cut_intervals: List[Interval]) -> Optional[Caption]:
    """Handle caption that partially overlaps with cut intervals."""
    # This is a more complex case - the caption spans a cut boundary
    # We need to split the caption or adjust it
    
    # For simplicity in MVP, if the caption overlaps a cut, 
    # try to use the portion that remains
    
    # Find the total removed duration before the caption start
    total_removed_before_start = 0.0
    for cut in cut_intervals:
        if cut.end <= caption.start:
            total_removed_before_start += cut.duration
    
    # Find the total removed duration before the caption end
    total_removed_before_end = 0.0
    for cut in cut_intervals:
        if cut.end <= caption.end:
            total_removed_before_end += cut.duration
    
    # Check if the caption overlaps any cut
    overlaps_cut = False
    for cut in cut_intervals:
        if (cut.start < caption.end and cut.end > caption.start):
            overlaps_cut = True
            break
    
    if overlaps_cut:
        # Try to find the keep portions
        # For now, skip captions that overlap cuts
        # A more sophisticated implementation would split captions
        return None
    
    new_start = caption.start - total_removed_before_start
    new_end = caption.end - total_removed_before_end
    
    if new_start >= new_end:
        return None
    
    return Caption(
        index=caption.index,
        start=new_start,
        end=new_end,
        text=caption.text,
        source_words=caption.source_words
    )


def remap_captions_batch(
    captions: List[Caption],
    cut_intervals: List[Interval]
) -> List[Caption]:
    """Remap multiple captions with batch optimization.
    
    Pre-computes the duration map once and uses it for all captions.
    
    Args:
        captions: List of Caption objects
        cut_intervals: List of cut Interval objects
        
    Returns:
        List of remapped Caption objects
    """
    # Sort and merge cuts for efficiency
    sorted_cuts = sort_intervals(cut_intervals)
    merged_cuts = merge_overlapping(sorted_cuts)
    
    # Build duration map
    duration_map = build_duration_map(merged_cuts)
    
    result = []
    for caption in captions:
        new_caption = _remap_single_caption(caption, merged_cuts, duration_map)
        if new_caption:
            result.append(new_caption)
    
    return result


def generate_srt(captions: List[Caption]) -> str:
    """Generate SRT format subtitles from captions.
    
    Args:
        captions: List of Caption objects (in edited timeline)
        
    Returns:
        SRT format string
    """
    srt_lines = []
    for caption in captions:
        srt_lines.append(caption.to_srt())
    return "".join(srt_lines)


def save_srt(captions: List[Caption], file_path: str) -> None:
    """Save captions as SRT file.
    
    Args:
        captions: List of Caption objects
        file_path: Path to save the SRT file
    """
    srt_content = generate_srt(captions)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)


def filter_captions_by_intervals(
    captions: List[Caption],
    keep_intervals: List[Interval]
) -> List[Caption]:
    """Filter captions to only include those within keep intervals.
    
    Args:
        captions: List of Caption objects
        keep_intervals: List of keep Interval objects
        
    Returns:
        List of Caption objects that are at least partially within keep intervals
    """
    if not keep_intervals:
        return captions
    
    # Merge keep intervals for efficiency
    merged_keeps = merge_overlapping(keep_intervals)
    
    result = []
    for caption in captions:
        # Check if caption overlaps with any keep interval
        for keep in merged_keeps:
            # Check overlap: caption overlaps if it's not completely before or after
            if caption.end > keep.start and caption.start < keep.end:
                result.append(caption)
                break
    
    return result


def reindex_captions(captions: List[Caption], start_index: int = 1) -> List[Caption]:
    """Re-index captions sequentially.
    
    Args:
        captions: List of Caption objects
        start_index: Starting index (default 1 for SRT)
        
    Returns:
        List of Caption objects with updated indices
    """
    for i, caption in enumerate(captions):
        caption.index = start_index + i
    return captions


def get_caption_words(caption: Caption, words: List[Word]) -> List[Word]:
    """Get the list of Word objects that contribute to a caption.
    
    Args:
        caption: Caption object
        words: List of all Word objects
        
    Returns:
        List of Word objects that are in caption.source_words
    """
    return [words[i] for i in caption.source_words if i < len(words)]
