"""
Tests for interval mathematics.

This module tests:
- Interval creation, validation, merging, clamping, padding
- Invert (complement) operations
- Timestamp mapping
- Edge cases
"""

import pytest
from resolve_autocut.models import Interval, FillerDetection, FillerType
from resolve_autocut.intervals import (
    sort_intervals, merge_overlapping, clamp_interval, clamp_intervals,
    pad_intervals, invert_intervals, map_timestamp_simple, build_duration_map,
    calculate_removed_before, validate_intervals, remap_intervals_to_edited,
    frames_to_seconds, seconds_to_frames, timecode_to_seconds, seconds_to_timecode,
    EPSILON
)


class TestInterval:
    """Tests for Interval class."""
    
    def test_create_basic(self):
        """Test basic interval creation."""
        interval = Interval(start=0.0, end=1.0)
        assert interval.start == 0.0
        assert interval.end == 1.0
        assert interval.duration == 1.0
    
    def test_create_with_floats(self):
        """Test interval creation with float values."""
        interval = Interval(start=1.5, end=2.75)
        assert interval.start == 1.5
        assert interval.end == 2.75
        assert abs(interval.duration - 1.25) < EPSILON
    
    def test_create_reversed_raises(self):
        """Test that reversed intervals raise an error."""
        with pytest.raises(ValueError):
            Interval(start=2.0, end=1.0)
    
    def test_contains(self):
        """Test interval contains method."""
        interval = Interval(start=1.0, end=3.0)
        
        assert interval.contains(1.0)
        assert interval.contains(2.0)
        assert interval.contains(2.999)
        assert not interval.contains(0.0)
        assert not interval.contains(3.0)
        assert not interval.contains(4.0)
    
    def test_overlaps(self):
        """Test interval overlap detection."""
        a = Interval(start=0.0, end=2.0)
        b = Interval(start=1.0, end=3.0)
        c = Interval(start=2.0, end=4.0)
        d = Interval(start=3.0, end=5.0)
        
        assert a.overlaps(b)  # Overlap
        assert b.overlaps(a)  # Symmetric
        assert not a.overlaps(c)  # NOT overlapping (adjacent at boundary)
        assert not a.overlaps(d)  # No overlap
    
    def test_clamp(self):
        """Test interval clamping."""
        interval = Interval(start=1.0, end=3.0)
        
        clamped = interval.clamp(2.0, 4.0)
        assert clamped.start == 2.0
        assert clamped.end == 3.0
        
        clamped = interval.clamp(0.0, 2.0)
        assert clamped.start == 1.0
        assert clamped.end == 2.0
        
        clamped = interval.clamp(0.0, 5.0)
        assert clamped.start == 1.0
        assert clamped.end == 3.0
        
        clamped = interval.clamp(4.0, 5.0)
        assert clamped.start == 4.0
        assert clamped.end == 4.0
    
    def test_pad(self):
        """Test interval padding."""
        interval = Interval(start=1.0, end=2.0)
        
        padded = interval.pad(0.1)
        assert abs(padded.start - 0.9) < EPSILON
        assert abs(padded.end - 2.1) < EPSILON
    
    def test_to_dict_and_from_dict(self):
        """Test interval serialization."""
        interval = Interval(start=1.5, end=3.5)
        data = interval.to_dict()
        
        assert data["start"] == 1.5
        assert data["end"] == 3.5
        
        restored = Interval.from_dict(data)
        assert restored.start == interval.start
        assert restored.end == interval.end
    
    def test_equality(self):
        """Test interval equality."""
        a = Interval(start=0.0, end=1.0)
        b = Interval(start=0.0, end=1.0)
        c = Interval(start=0.0, end=2.0)
        
        assert a == b
        assert a != c
        assert a != "not an interval"
    
    def test_less_than(self):
        """Test interval comparison."""
        a = Interval(start=0.0, end=1.0)
        b = Interval(start=1.0, end=2.0)
        c = Interval(start=0.5, end=1.5)
        
        assert a < b
        assert a < c
        assert not (b < a)


class TestSortIntervals:
    """Tests for sorting intervals."""
    
    def test_empty_list(self):
        """Test sorting empty list."""
        assert sort_intervals([]) == []
    
    def test_single_interval(self):
        """Test sorting single interval."""
        intervals = [Interval(start=1.0, end=2.0)]
        assert sort_intervals(intervals) == intervals
    
    def test_multiple_intervals(self):
        """Test sorting multiple intervals."""
        intervals = [
            Interval(start=3.0, end=4.0),
            Interval(start=1.0, end=2.0),
            Interval(start=2.0, end=3.0)
        ]
        sorted_intervals = sort_intervals(intervals)
        
        assert sorted_intervals[0].start == 1.0
        assert sorted_intervals[1].start == 2.0
        assert sorted_intervals[2].start == 3.0


class TestMergeOverlapping:
    """Tests for merging overlapping intervals."""
    
    def test_empty_list(self):
        """Test merging empty list."""
        assert merge_overlapping([]) == []
    
    def test_single_interval(self):
        """Test merging single interval."""
        intervals = [Interval(start=1.0, end=2.0)]
        assert merge_overlapping(intervals) == intervals
    
    def test_no_overlap(self):
        """Test merging non-overlapping intervals."""
        intervals = [
            Interval(start=0.0, end=1.0),
            Interval(start=2.0, end=3.0),
            Interval(start=4.0, end=5.0)
        ]
        merged = merge_overlapping(intervals)
        assert len(merged) == 3
        assert merged[0].start == 0.0
        assert merged[1].start == 2.0
        assert merged[2].start == 4.0
    
    def test_overlapping(self):
        """Test merging overlapping intervals."""
        intervals = [
            Interval(start=0.0, end=2.0),
            Interval(start=1.0, end=3.0)
        ]
        merged = merge_overlapping(intervals)
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end == 3.0
    
    def test_adjacent(self):
        """Test merging adjacent intervals."""
        intervals = [
            Interval(start=0.0, end=1.0),
            Interval(start=1.0, end=2.0)
        ]
        merged = merge_overlapping(intervals)
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end == 2.0
    
    def test_multiple_merges(self):
        """Test merging multiple overlapping intervals."""
        intervals = [
            Interval(start=0.0, end=1.5),
            Interval(start=1.0, end=2.0),
            Interval(start=3.0, end=4.0),
            Interval(start=3.5, end=5.0)
        ]
        merged = merge_overlapping(intervals)
        assert len(merged) == 2
        assert merged[0].start == 0.0
        assert merged[0].end == 2.0
        assert merged[1].start == 3.0
        assert merged[1].end == 5.0
    
    def test_unsorted_input(self):
        """Test merging unsorted intervals."""
        intervals = [
            Interval(start=3.0, end=4.0),
            Interval(start=1.0, end=2.0),
            Interval(start=0.0, end=1.5)
        ]
        merged = merge_overlapping(intervals)
        assert len(merged) == 2
        assert merged[0].start == 0.0
        assert merged[0].end == 2.0
        assert merged[1].start == 3.0
        assert merged[1].end == 4.0


class TestClampIntervals:
    """Tests for clamping intervals."""
    
    def test_clamp_single(self):
        """Test clamping a single interval."""
        interval = Interval(start=-1.0, end=1.5)
        clamped = clamp_interval(interval, 0.0, 1.0)
        
        assert clamped is not None
        assert clamped.start == 0.0
        assert clamped.end == 1.0
    
    def test_clamp_multiple(self):
        """Test clamping multiple intervals."""
        intervals = [
            Interval(start=-1.0, end=0.5),
            Interval(start=0.5, end=1.5),
            Interval(start=2.0, end=3.0)
        ]
        clamped = clamp_intervals(intervals, 0.0, 2.0)
        
        assert len(clamped) == 2
        assert clamped[0].start == 0.0
        assert clamped[0].end == 0.5
        assert clamped[1].start == 0.5
        assert clamped[1].end == 1.5
    
    def test_clamp_completely_outside(self):
        """Test clamping intervals completely outside bounds."""
        intervals = [
            Interval(start=-2.0, end=-1.0),
            Interval(start=3.0, end=4.0)
        ]
        clamped = clamp_intervals(intervals, 0.0, 2.0)
        
        assert len(clamped) == 0


class TestPadIntervals:
    """Tests for padding intervals."""
    
    def test_pad_single(self):
        """Test padding a single interval."""
        intervals = [Interval(start=1.0, end=2.0)]
        padded = pad_intervals(intervals, 0.1, 0.0, 3.0)
        
        assert len(padded) == 1
        assert abs(padded[0].start - 0.9) < EPSILON
        assert abs(padded[0].end - 2.1) < EPSILON
    
    def test_pad_clamped(self):
        """Test padding with clamping."""
        intervals = [Interval(start=0.0, end=0.5)]
        padded = pad_intervals(intervals, 0.1, 0.0, 1.0)
        
        assert len(padded) == 1
        assert padded[0].start == 0.0  # Clamped to min
        assert abs(padded[0].end - 0.6) < EPSILON


class TestInvertIntervals:
    """Tests for inverting intervals (finding keep intervals)."""
    
    def test_empty_cuts(self):
        """Test inverting with no cuts."""
        cuts = []
        keeps = invert_intervals(cuts, 10.0)
        
        assert len(keeps) == 1
        assert keeps[0].start == 0.0
        assert keeps[0].end == 10.0
    
    def test_single_cut(self):
        """Test inverting with a single cut."""
        cuts = [Interval(start=2.0, end=4.0)]
        keeps = invert_intervals(cuts, 10.0)
        
        assert len(keeps) == 2
        assert keeps[0].start == 0.0
        assert keeps[0].end == 2.0
        assert keeps[1].start == 4.0
        assert keeps[1].end == 10.0
    
    def test_multiple_cuts(self):
        """Test inverting with multiple cuts."""
        cuts = [
            Interval(start=1.0, end=2.0),
            Interval(start=5.0, end=7.0),
            Interval(start=8.0, end=9.0)
        ]
        keeps = invert_intervals(cuts, 10.0)
        
        assert len(keeps) == 4
        assert keeps[0].start == 0.0
        assert keeps[0].end == 1.0
        assert keeps[1].start == 2.0
        assert keeps[1].end == 5.0
        assert keeps[2].start == 7.0
        assert keeps[2].end == 8.0
        assert keeps[3].start == 9.0
        assert keeps[3].end == 10.0
    
    def test_cut_at_start(self):
        """Test inverting with cut at start."""
        cuts = [Interval(start=0.0, end=2.0)]
        keeps = invert_intervals(cuts, 10.0)
        
        assert len(keeps) == 1
        assert keeps[0].start == 2.0
        assert keeps[0].end == 10.0
    
    def test_cut_at_end(self):
        """Test inverting with cut at end."""
        cuts = [Interval(start=8.0, end=10.0)]
        keeps = invert_intervals(cuts, 10.0)
        
        assert len(keeps) == 1
        assert keeps[0].start == 0.0
        assert keeps[0].end == 8.0
    
    def test_cut_full_duration(self):
        """Test inverting with cut covering full duration."""
        cuts = [Interval(start=0.0, end=10.0)]
        keeps = invert_intervals(cuts, 10.0)
        
        assert len(keeps) == 0
    
    def test_overlapping_cuts(self):
        """Test inverting with overlapping cuts (should be merged)."""
        cuts = [
            Interval(start=1.0, end=3.0),
            Interval(start=2.0, end=4.0)
        ]
        keeps = invert_intervals(cuts, 10.0)
        
        assert len(keeps) == 2
        assert keeps[0].start == 0.0
        assert keeps[0].end == 1.0
        assert keeps[1].start == 4.0
        assert keeps[1].end == 10.0


class TestTimestampMapping:
    """Tests for timestamp mapping after cuts."""
    
    def test_no_cuts(self):
        """Test mapping with no cuts."""
        timestamp = 10.0
        cuts = []
        mapped = map_timestamp_simple(timestamp, cuts)
        
        assert mapped == 10.0
    
    def test_single_cut_before(self):
        """Test mapping with a cut before the timestamp."""
        timestamp = 20.0
        cuts = [Interval(start=10.0, end=10.5)]
        mapped = map_timestamp_simple(timestamp, cuts)
        
        assert abs(mapped - 19.5) < EPSILON
    
    def test_single_cut_after(self):
        """Test mapping with a cut after the timestamp."""
        timestamp = 5.0
        cuts = [Interval(start=10.0, end=10.5)]
        mapped = map_timestamp_simple(timestamp, cuts)
        
        assert mapped == 5.0
    
    def test_timestamp_in_cut(self):
        """Test mapping a timestamp that falls in a cut."""
        timestamp = 10.25
        cuts = [Interval(start=10.0, end=10.5)]
        mapped = map_timestamp_simple(timestamp, cuts)
        
        assert mapped is None
    
    def test_multiple_cuts(self):
        """Test mapping with multiple cuts before timestamp."""
        timestamp = 20.0
        cuts = [
            Interval(start=5.0, end=5.5),
            Interval(start=10.0, end=10.5),
            Interval(start=15.0, end=15.5)
        ]
        mapped = map_timestamp_simple(timestamp, cuts)
        
        # Total removed: 0.5 + 0.5 + 0.5 = 1.5
        assert abs(mapped - 18.5) < EPSILON
    
    def test_cut_at_start(self):
        """Test mapping with cut at start."""
        timestamp = 1.0
        cuts = [Interval(start=0.0, end=0.5)]
        mapped = map_timestamp_simple(timestamp, cuts)
        
        assert abs(mapped - 0.5) < EPSILON
    
    def test_cut_at_timestamp(self):
        """Test mapping with cut ending at timestamp."""
        timestamp = 10.5
        cuts = [Interval(start=10.0, end=10.5)]
        mapped = map_timestamp_simple(timestamp, cuts)
        
        # The timestamp is at the end of the cut, so it should be removed
        # But our interval is [start, end), so 10.5 is NOT in [10.0, 10.5)
        # So it should map to 10.0
        assert abs(mapped - 10.0) < EPSILON


class TestBuildDurationMap:
    """Tests for building duration map."""
    
    def test_empty(self):
        """Test with no cuts."""
        cuts = []
        duration_map = build_duration_map(cuts)
        
        assert duration_map == []
    
    def test_single_cut(self):
        """Test with a single cut."""
        cuts = [Interval(start=5.0, end=6.0)]
        duration_map = build_duration_map(cuts)
        
        assert len(duration_map) == 1
        assert duration_map[0][0] == 5.0
        assert duration_map[0][1] == 1.0
    
    def test_overlapping_cuts_merged(self):
        """Test that overlapping cuts are merged in duration map."""
        cuts = [
            Interval(start=5.0, end=6.0),
            Interval(start=5.5, end=7.0)
        ]
        duration_map = build_duration_map(cuts)
        
        assert len(duration_map) == 1
        assert duration_map[0][0] == 5.0
        assert duration_map[0][1] == 2.0  # Merged duration


class TestValidateIntervals:
    """Tests for interval validation."""
    
    def test_valid_intervals(self):
        """Test validation of valid intervals."""
        intervals = [
            Interval(start=0.0, end=1.0),
            Interval(start=2.0, end=3.0)
        ]
        valid, errors = validate_intervals(intervals, 10.0)
        
        assert valid
        assert len(errors) == 0
    
    def test_negative_start(self):
        """Test detection of negative start."""
        intervals = [Interval(start=-1.0, end=1.0)]
        valid, errors = validate_intervals(intervals, 10.0)
        
        assert not valid
        assert len(errors) > 0
    
    def test_exceeds_duration(self):
        """Test detection of interval exceeding duration."""
        intervals = [Interval(start=0.0, end=20.0)]
        valid, errors = validate_intervals(intervals, 10.0)
        
        assert not valid
        assert len(errors) > 0
    
    def test_reversed_interval(self):
        """Test detection of reversed interval."""
        # Note: Interval constructor should raise an error for reversed intervals
        # But we can test with start == end
        intervals = [Interval(start=1.0, end=1.0)]
        valid, errors = validate_intervals(intervals, 10.0)
        
        assert not valid
    
    def test_overlapping_intervals(self):
        """Test detection of overlapping intervals."""
        intervals = [
            Interval(start=0.0, end=3.0),
            Interval(start=2.0, end=4.0)
        ]
        valid, errors = validate_intervals(intervals, 10.0)
        
        assert not valid


class TestRemapIntervalsToEdited:
    """Tests for remapping intervals to edited timeline."""
    
    def test_remap_no_cuts(self):
        """Test remapping with no cuts."""
        source_intervals = [Interval(start=0.0, end=1.0)]
        cuts = []
        remapped = remap_intervals_to_edited(source_intervals, cuts)
        
        assert len(remapped) == 1
        assert remapped[0].start == 0.0
        assert remapped[0].end == 1.0
    
    def test_remap_with_cut_before(self):
        """Test remapping with cut before interval."""
        source_intervals = [Interval(start=10.0, end=12.0)]
        cuts = [Interval(start=5.0, end=6.0)]
        remapped = remap_intervals_to_edited(source_intervals, cuts)
        
        assert len(remapped) == 1
        assert abs(remapped[0].start - 9.0) < EPSILON
        assert abs(remapped[0].end - 11.0) < EPSILON
    
    def test_remap_interval_in_cut(self):
        """Test remapping interval completely in cut."""
        source_intervals = [Interval(start=5.0, end=6.0)]
        cuts = [Interval(start=5.0, end=6.0)]
        remapped = remap_intervals_to_edited(source_intervals, cuts)
        
        assert len(remapped) == 0

    def test_remap_interval_starting_inside_cut(self):
        remapped = remap_intervals_to_edited(
            [Interval(start=5.5, end=7.0)], [Interval(start=5.0, end=6.0)]
        )
        assert remapped == [Interval(start=5.0, end=6.0)]

    def test_remap_interval_spanning_multiple_cuts(self):
        remapped = remap_intervals_to_edited(
            [Interval(start=1.0, end=9.0)],
            [Interval(start=2.0, end=3.0), Interval(start=5.0, end=7.0)],
        )
        assert remapped == [Interval(start=1.0, end=6.0)]


class TestFramesAndSeconds:
    """Tests for frame/second conversions."""
    
    def test_frames_to_seconds(self):
        """Test frames to seconds conversion."""
        assert frames_to_seconds(24, 24.0) == 1.0
        assert frames_to_seconds(48, 24.0) == 2.0
        assert frames_to_seconds(12, 24.0) == 0.5
    
    def test_seconds_to_frames(self):
        """Test seconds to frames conversion."""
        assert seconds_to_frames(1.0, 24.0) == 24
        assert seconds_to_frames(2.0, 24.0) == 48
        assert seconds_to_frames(0.5, 24.0) == 12


class TestTimecodeConversions:
    """Tests for timecode conversions."""
    
    def test_seconds_to_timecode_24fps(self):
        """Test seconds to timecode at 24fps."""
        # 1 second = 24 frames
        tc = seconds_to_timecode(1.0, 24.0)
        assert tc == "00:00:01:00"
        
        # 1 minute = 60 seconds = 1440 frames
        tc = seconds_to_timecode(60.0, 24.0)
        assert tc == "00:01:00:00"
    
    def test_timecode_to_seconds_24fps(self):
        """Test timecode to seconds at 24fps."""
        seconds = timecode_to_seconds("00:00:01:00", 24.0)
        assert abs(seconds - 1.0) < EPSILON
        
        seconds = timecode_to_seconds("00:01:00:00", 24.0)
        assert abs(seconds - 60.0) < EPSILON
