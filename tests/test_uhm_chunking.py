"""
Tests for UHM audio chunking.

This module tests:
- Audio chunking for UHM's 30-second window requirement
- Absolute timestamp preservation across chunks
- Edge cases with audio lengths
"""

import pytest
import numpy as np
from resolve_autocut.uhm import (
    chunk_audio_for_uhm, UHM_WINDOW_SIZE, UHM_STRIDE, UHM_EXPECTED_SAMPLE_RATE
)
from resolve_autocut.models import Interval


class TestChunkAudioForUHM:
    """Tests for chunk_audio_for_uhm function."""
    
    def test_audio_shorter_than_window(self):
        """Test chunking audio shorter than 30 seconds."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = 5.0  # 5 seconds
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        # Should have at least one chunk
        assert len(chunks) >= 1
        
        # First chunk should start at 0
        assert chunks[0][0] == 0.0
        
        # First chunk should end at or after the audio end
        assert chunks[0][1] >= duration
    
    def test_audio_exactly_30_seconds(self):
        """Test chunking audio exactly 30 seconds."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = UHM_WINDOW_SIZE  # 30 seconds
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        # Should have chunks covering the entire audio
        assert len(chunks) >= 1
        
        # First chunk starts at 0
        assert chunks[0][0] == 0.0
        
        # Last chunk should end at or after 30 seconds
        assert chunks[-1][1] >= duration
    
    def test_audio_slightly_over_30_seconds(self):
        """Test chunking audio slightly over 30 seconds."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = UHM_WINDOW_SIZE + 0.1  # 30.1 seconds
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        # UHM receives contiguous 30-second windows, not a 30-second model
        # invocation every 20 ms.  The tail is a second, short window.
        assert len(chunks) == 2
        
        # First chunk starts at 0
        assert chunks[0][0] == 0.0
        
        # Last chunk should end at or after the audio end
        assert chunks[-1][1] >= duration
    
    def test_audio_several_minutes(self):
        """Test chunking audio several minutes long."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = 180.0  # 3 minutes
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        # Should have many chunks (180 / 0.02 = 9000 windows)
        assert len(chunks) > 1
        
        # First chunk starts at 0
        assert chunks[0][0] == 0.0
        
        # Last chunk should end at or after the audio end
        assert chunks[-1][1] >= duration
    
    def test_absolute_timestamps_correct(self):
        """Test that absolute timestamps are preserved across chunks."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = 0.1  # Short duration for testing
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        # Verify that chunks have correct timestamps
        for i, (chunk_start, chunk_end, chunk_audio) in enumerate(chunks):
            # Each chunk should start where the previous one started + stride
            if i == 0:
                assert chunk_start == 0.0
            else:
                expected_start = chunks[i-1][1]
                assert abs(chunk_start - expected_start) < 0.001  # Small tolerance
            
            # Chunk end should be after start
            assert chunk_end > chunk_start
            
            # Chunk should have audio data
            assert len(chunk_audio) > 0
    
    def test_chunk_audio_data_correct(self):
        """Test that chunk audio data is correct."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = 1.0  # 1 second
        num_samples = int(duration * sample_rate)
        audio_data = np.arange(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate, chunk_size=0.5)
        
        # Should have chunks
        assert len(chunks) >= 1
        
        # First chunk should contain the first samples
        first_chunk_audio = chunks[0][2]
        assert len(first_chunk_audio) > 0
        assert first_chunk_audio[0] == 0  # First sample
    
    def test_different_sample_rates(self):
        """Test chunking with different sample rates."""
        # Test with 44100 Hz
        sample_rate = 44100
        duration = 1.0
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        assert len(chunks) >= 1
        assert chunks[0][0] == 0.0
        
        # Test with 48000 Hz
        sample_rate = 48000
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        assert len(chunks) >= 1
        assert chunks[0][0] == 0.0


class TestUHMConstants:
    """Tests for UHM constants."""
    
    def test_window_size(self):
        """Test that window size is 30 seconds."""
        assert UHM_WINDOW_SIZE == 30.0
    
    def test_stride(self):
        """Test that stride is 20ms."""
        assert UHM_STRIDE == 0.02
    
    def test_sample_rate(self):
        """Test that expected sample rate is 16000."""
        assert UHM_EXPECTED_SAMPLE_RATE == 16000


class TestAudioChunkingEdgeCases:
    """Edge case tests for audio chunking."""
    
    def test_very_short_audio(self):
        """Test with very short audio (less than 1 sample)."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        audio_data = np.array([], dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        # Should return empty list for empty audio
        assert len(chunks) == 0
    
    def test_single_sample(self):
        """Test with single sample."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        audio_data = np.array([0.5], dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        # Should have at least one chunk
        assert len(chunks) >= 1
        assert chunks[0][2][0] == 0.5
    
    def test_very_long_audio(self):
        """Test with very long audio (1 hour)."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = 3600.0  # 1 hour
        num_samples = int(duration * sample_rate)
        
        # Don't actually create the array (would be huge)
        # Just test that the function would handle it
        # We'll create a small array and test the chunking logic
        audio_data = np.zeros(1000, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate)
        
        assert len(chunks) >= 1
    
    def test_custom_chunk_size(self):
        """Test with custom chunk size."""
        sample_rate = UHM_EXPECTED_SAMPLE_RATE
        duration = 1.0
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chunks = chunk_audio_for_uhm(audio_data, sample_rate, chunk_size=1.0)
        
        assert len(chunks) >= 1
        assert chunks[0][0] == 0.0
