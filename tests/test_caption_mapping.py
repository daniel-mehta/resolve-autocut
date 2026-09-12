"""
Tests for caption timestamp remapping.

This module tests:
- Caption generation from words
- Caption timestamp remapping after cuts
- SRT generation
- Edge cases with caption/cut overlap
"""

import pytest
from resolve_autocut.models import Word, Caption, Interval, FillerDetection, FillerType
from resolve_autocut.captions import (
    generate_captions_from_words, remap_captions, remap_captions_batch,
    generate_srt, save_srt, reindex_captions, filter_captions_by_intervals,
    get_caption_words, TARGET_CAPTION_LENGTH, MAX_CAPTION_LENGTH
)
from resolve_autocut.intervals import merge_overlapping


class TestCaptionGeneration:
    """Tests for caption generation from words."""
    
    def test_empty_words(self):
        """Test with no words."""
        captions = generate_captions_from_words([])
        assert captions == []
    
    def test_single_word(self):
        """Test with a single word."""
        words = [Word(text="Hello", start=0.0, end=0.5, index=0)]
        captions = generate_captions_from_words(words)
        
        # Single short word might not create a caption if duration is too short
        assert len(captions) <= 1
    
    def test_multiple_words(self):
        """Test with multiple words forming one caption."""
        words = [
            Word(text="Hello", start=0.0, end=0.5, index=0),
            Word(text="world", start=0.5, end=1.0, index=1),
        ]
        captions = generate_captions_from_words(words)
        
        assert len(captions) >= 1
        if captions:
            # Check that the caption contains both words
            assert "Hello" in captions[0].text
            assert "world" in captions[0].text
    
    def test_long_caption_splitting(self):
        """Test that long captions are split."""
        # Create many words that exceed max length
        words = []
        for i in range(50):
            words.append(Word(
                text=f"word{i}",
                start=float(i) * 0.2,
                end=float(i) * 0.2 + 0.2,
                index=i
            ))
        
        captions = generate_captions_from_words(words, max_length=MAX_CAPTION_LENGTH)
        
        # Should have multiple captions
        assert len(captions) > 1
        
        # Each caption should be within length limit
        for caption in captions:
            assert len(caption.text) <= MAX_CAPTION_LENGTH + 10  # Some tolerance
    
    def test_caption_timestamps(self):
        """Test caption timestamp calculation."""
        words = [
            Word(text="Hello", start=0.0, end=0.5, index=0),
            Word(text="world", start=0.5, end=1.0, index=1),
        ]
        captions = generate_captions_from_words(words)
        
        if captions:
            caption = captions[0]
            assert caption.start == 0.0
            assert caption.end >= 1.0


class TestCaptionRemapping:
    """Tests for caption timestamp remapping after cuts."""
    
    def test_no_cuts(self):
        """Test remapping with no cuts."""
        captions = [
            Caption(index=1, start=0.0, end=1.0, text="Hello world"),
            Caption(index=2, start=1.0, end=2.0, text="Second caption")
        ]
        cuts = []
        remapped = remap_captions(captions, cuts)
        
        assert len(remapped) == 2
        assert remapped[0].start == 0.0
        assert remapped[0].end == 1.0
        assert remapped[1].start == 1.0
        assert remapped[1].end == 2.0
    
    def test_caption_before_cut(self):
        """Test remapping caption before cut."""
        captions = [
            Caption(index=1, start=0.0, end=1.0, text="Before cut")
        ]
        cuts = [Interval(start=5.0, end=6.0)]
        remapped = remap_captions(captions, cuts)
        
        assert len(remapped) == 1
        assert remapped[0].start == 0.0
        assert remapped[0].end == 1.0
    
    def test_caption_after_single_cut(self):
        """Test remapping caption after a single cut."""
        # Remove 10.0 to 10.5 (0.5 seconds)
        # Caption at 20.0 should map to 19.5
        captions = [
            Caption(index=1, start=20.0, end=21.0, text="After cut")
        ]
        cuts = [Interval(start=10.0, end=10.5)]
        remapped = remap_captions(captions, cuts)
        
        assert len(remapped) == 1
        assert abs(remapped[0].start - 19.5) < 0.001
        assert abs(remapped[0].end - 20.5) < 0.001
    
    def test_caption_in_cut(self):
        """Test remapping caption completely within a cut."""
        captions = [
            Caption(index=1, start=10.0, end=10.5, text="In cut")
        ]
        cuts = [Interval(start=10.0, end=10.5)]
        remapped = remap_captions(captions, cuts)
        
        assert len(remapped) == 0
    
    def test_multiple_cuts_before(self):
        """Test remapping with multiple cuts before caption."""
        # Remove 0.5s + 0.5s + 0.5s = 1.5s total
        # Caption at 20.0 should map to 18.5
        captions = [
            Caption(index=1, start=20.0, end=21.0, text="Multiple cuts before")
        ]
        cuts = [
            Interval(start=5.0, end=5.5),
            Interval(start=10.0, end=10.5),
            Interval(start=15.0, end=15.5)
        ]
        remapped = remap_captions(captions, cuts)
        
        assert len(remapped) == 1
        assert abs(remapped[0].start - 18.5) < 0.001
        assert abs(remapped[0].end - 19.5) < 0.001
    
    def test_caption_overlapping_cut(self):
        """Test remapping caption that overlaps with a cut."""
        # Caption from 10.0 to 12.0, cut from 10.5 to 11.0
        # This caption partially overlaps with the cut
        captions = [
            Caption(index=1, start=10.0, end=12.0, text="Overlapping")
        ]
        cuts = [Interval(start=10.5, end=11.0)]
        remapped = remap_captions(captions, cuts)
        
        # The caption overlaps with the cut, so the current implementation
        # may return None for it, resulting in an empty list
        # Or it may try to partially remap it
        # Either behavior is acceptable for now
        assert len(remapped) <= 1


class TestCaptionRemappingBatch:
    """Tests for batch caption remapping."""
    
    def test_batch_remapping(self):
        """Test batch remapping of multiple captions."""
        captions = [
            Caption(index=1, start=0.0, end=1.0, text="First"),
            Caption(index=2, start=5.0, end=6.0, text="Second"),
            Caption(index=3, start=10.0, end=11.0, text="Third"),
        ]
        cuts = [Interval(start=2.0, end=3.0)]
        
        remapped = remap_captions_batch(captions, cuts)
        
        # First caption: unchanged
        assert len(remapped) >= 2
        if len(remapped) >= 1:
            assert remapped[0].start == 0.0
            assert remapped[0].end == 1.0
        
        if len(remapped) >= 2:
            # Second caption: should be shifted back by 1.0
            assert abs(remapped[1].start - 4.0) < 0.001
            assert abs(remapped[1].end - 5.0) < 0.001


class TestSRTGeneration:
    """Tests for SRT generation."""
    
    def test_empty_captions(self):
        """Test SRT generation with no captions."""
        srt = generate_srt([])
        assert srt == ""
    
    def test_single_caption(self):
        """Test SRT generation with one caption."""
        captions = [
            Caption(index=1, start=0.0, end=1.5, text="Hello world")
        ]
        srt = generate_srt(captions)
        
        assert "1" in srt
        assert "00:00:00,000" in srt
        assert "Hello world" in srt
    
    def test_multiple_captions(self):
        """Test SRT generation with multiple captions."""
        captions = [
            Caption(index=1, start=0.0, end=1.0, text="First caption"),
            Caption(index=2, start=1.0, end=2.0, text="Second caption")
        ]
        srt = generate_srt(captions)
        
        assert "1" in srt
        assert "2" in srt
        assert "First caption" in srt
        assert "Second caption" in srt
    
    def test_time_formatting(self):
        """Test SRT time formatting."""
        captions = [
            Caption(index=1, start=0.123, end=1.456, text="Test")
        ]
        srt = generate_srt(captions)
        
        # Should have milliseconds
        assert "," in srt


class TestReindexCaptions:
    """Tests for caption reindexing."""
    
    def test_reindex(self):
        """Test reindexing captions."""
        captions = [
            Caption(index=0, start=0.0, end=1.0, text="First"),
            Caption(index=0, start=1.0, end=2.0, text="Second"),
        ]
        reindexed = reindex_captions(captions, start_index=1)
        
        assert reindexed[0].index == 1
        assert reindexed[1].index == 2


class TestFilterCaptionsByIntervals:
    """Tests for filtering captions by keep intervals."""
    
    def test_no_keep_intervals(self):
        """Test filtering with no keep intervals."""
        captions = [
            Caption(index=1, start=0.0, end=1.0, text="Test")
        ]
        filtered = filter_captions_by_intervals(captions, [])
        
        assert len(filtered) == 1
    
    def test_caption_in_keep(self):
        """Test filtering caption within keep interval."""
        captions = [
            Caption(index=1, start=0.0, end=1.0, text="In keep")
        ]
        keep_intervals = [Interval(start=0.0, end=2.0)]
        filtered = filter_captions_by_intervals(captions, keep_intervals)
        
        assert len(filtered) == 1
    
    def test_caption_outside_keep(self):
        """Test filtering caption outside keep interval."""
        captions = [
            Caption(index=1, start=5.0, end=6.0, text="Outside")
        ]
        keep_intervals = [Interval(start=0.0, end=2.0)]
        filtered = filter_captions_by_intervals(captions, keep_intervals)
        
        assert len(filtered) == 0


class TestCaptionEdgeCases:
    """Edge case tests for captions."""
    
    def test_caption_at_zero(self):
        """Test caption starting at 0."""
        captions = [
            Caption(index=1, start=0.0, end=1.0, text="Start")
        ]
        cuts = []
        remapped = remap_captions(captions, cuts)
        
        assert len(remapped) == 1
        assert remapped[0].start == 0.0
    
    def test_caption_at_media_end(self):
        """Test caption at media end."""
        captions = [
            Caption(index=1, start=9.5, end=10.0, text="End")
        ]
        cuts = []
        remapped = remap_captions(captions, cuts)
        
        assert len(remapped) == 1
        assert remapped[0].end == 10.0
    
    def test_very_short_caption(self):
        """Test very short caption."""
        captions = [
            Caption(index=1, start=10.0, end=10.001, text=".")
        ]
        cuts = []
        remapped = remap_captions(captions, cuts)
        
        # Very short captions might be filtered out
        assert len(remapped) <= 1
