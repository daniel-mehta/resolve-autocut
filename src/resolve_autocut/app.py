"""
Main application logic for Resolve AutoCut.

This module provides the core analysis pipeline that:
1. Inspects media files
2. Extracts audio
3. Detects filler words
4. Transcribes audio
5. Generates captions
6. Maps timestamps after cuts
7. Exports FCPXML timeline
8. Exports SRT subtitles
9. Exports JSON analysis
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from .models import (
    MediaInfo, Interval, FillerDetection, Word, Caption,
    AnalysisResult, FillerType, DEFAULT_REMOVABLE_FILLERS
)
from .media import (
    inspect_media, extract_audio, validate_media,
    MediaError, FFmpegNotFoundError
)
from .uhm import (
    UHMModel, FillerDetector, detect_fillers, filter_detections,
    UHMError, ModelDownloadError
)
from .transcription import (
    WhisperTranscriber, convert_to_words, get_full_transcript,
    TranscriptionError, MLXNotAvailableError
)
from .intervals import (
    sort_intervals, merge_overlapping, invert_intervals,
    map_timestamp_simple, build_duration_map, pad_intervals,
    remap_intervals_to_edited, validate_intervals
)
from .captions import (
    generate_captions_from_words, remap_captions, generate_srt,
    save_srt, reindex_captions
)
from .timeline import (
    generate_fcpxml, generate_fcpxml_from_analysis, snap_cut_intervals_to_frames,
)


logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Exception raised for analysis pipeline errors."""
    pass


class AnalysisCancelled(AnalysisError):
    """Raised when a caller requests cooperative pipeline cancellation."""


class ResolveAutoCut:
    """Main application class for Resolve AutoCut.
    
    This class orchestrates the complete analysis pipeline.
    """
    
    def __init__(
        self,
        cut_padding: float = 0.05,
        removable_fillers: set = DEFAULT_REMOVABLE_FILLERS,
        whisper_model: str = "base",
        confidence_threshold: float = 0.75,
        min_filler_duration: float = 0.1,
        max_filler_duration: float = 5.0
    ):
        """Initialize Resolve AutoCut.
        
        Args:
            cut_padding: Padding to add to cut intervals (seconds)
            removable_fillers: Set of FillerType to mark as enabled by default
            whisper_model: Whisper model size
            confidence_threshold: Minimum confidence for filler detection
            min_filler_duration: Minimum duration for a filler detection
            max_filler_duration: Maximum duration for a filler detection
        """
        if cut_padding < 0:
            raise ValueError("Cut padding cannot be negative")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if min_filler_duration < 0 or max_filler_duration < min_filler_duration:
            raise ValueError("Invalid filler duration bounds")
        if not whisper_model:
            raise ValueError("Whisper model cannot be empty")
        self.cut_padding = cut_padding
        self.removable_fillers = set(removable_fillers)
        self.whisper_model = whisper_model
        self.confidence_threshold = confidence_threshold
        self.min_filler_duration = min_filler_duration
        self.max_filler_duration = max_filler_duration
        
        # Cache models
        self._uhm_model = None
        self._whisper_transcriber = None
        
        # Results cache
        self._last_analysis: Optional[AnalysisResult] = None
    
    def analyze(
        self,
        media_path: str,
        progress_callback=None
    ) -> AnalysisResult:
        """Run complete analysis on a media file.
        
        This is the main pipeline method that:
        1. Validates the media file
        2. Extracts audio
        3. Detects filler words
        4. Transcribes audio
        5. Generates captions
        6. Calculates keep intervals
        7. Remaps captions
        
        Args:
            media_path: Path to the media file to analyze
            progress_callback: Optional function to report progress (0-100)
            
        Returns:
            AnalysisResult with complete analysis
            
        Raises:
            AnalysisError: If any step in the pipeline fails
        """
        import tempfile
        
        logger.info(f"Starting analysis of {media_path}")
        
        # Step 1: Validate media
        if progress_callback:
            progress_callback({"step": "validating", "percent": 0})
        
        is_valid, error_msg = validate_media(media_path)
        if not is_valid:
            raise AnalysisError(f"Media validation failed: {error_msg}")
        
        # Step 2: Inspect media
        if progress_callback:
            progress_callback({"step": "inspecting", "percent": 5})
        
        media_info = inspect_media(media_path)
        logger.info(f"Media info: {media_info}")
        
        # Step 3: Extract audio for analysis
        if progress_callback:
            progress_callback({"step": "extracting_audio", "percent": 10})
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = audio_file.name
        
        try:
            extract_audio(
                media_path,
                audio_path,
                target_sample_rate=16000,
                target_channels=1,
                # NamedTemporaryFile creates an empty path; extraction must
                # replace it rather than treating it as a cached WAV.
                force=True,
            )
            
            # Step 4: Detect fillers with UHM
            if progress_callback:
                progress_callback({"step": "detecting_fillers", "percent": 15})
            
            filler_detections = self._detect_fillers(
                audio_path,
                (lambda fraction: progress_callback({
                    "step": "detecting_fillers",
                    "percent": 15 + 15 * fraction,
                })) if progress_callback else None,
            )
            logger.info(f"Detected {len(filler_detections)} filler regions")
            
            # Filter detections
            filler_detections = filter_detections(
                filler_detections,
                min_duration=self.min_filler_duration,
                max_duration=self.max_filler_duration,
                min_confidence=self.confidence_threshold
            )
            logger.info(f"Filtered to {len(filler_detections)} filler detections")
            
            # Step 5: Transcribe audio
            if progress_callback:
                progress_callback({"step": "transcribing", "percent": 30})
            
            full_transcript, words = self._transcribe(audio_path)
            logger.info(f"Transcription complete: {len(words)} words")
            
            # Step 6: Generate captions from words
            if progress_callback:
                progress_callback({"step": "generating_captions", "percent": 50})
            
            captions = generate_captions_from_words(words)
            logger.info(f"Generated {len(captions)} captions")
            
            # Step 7: Calculate cut intervals from enabled detections
            if progress_callback:
                progress_callback({"step": "calculating_cuts", "percent": 60})
            
            # Get enabled detections
            enabled_detections = [
                d for d in filler_detections if d.enabled
            ]
            
            # Convert to intervals with padding
            cut_intervals = []
            for det in enabled_detections:
                padded = det.pad(self.cut_padding)
                # Clamp to media bounds
                interval = padded.interval.clamp(0, media_info.duration)
                cut_intervals.append(interval)
            
            # Merge overlapping cuts
            merged_cuts = snap_cut_intervals_to_frames(
                media_info, merge_overlapping(cut_intervals)
            )
            
            # Calculate keep intervals
            keep_intervals = invert_intervals(merged_cuts, media_info.duration)
            
            logger.info(f"Calculated {len(merged_cuts)} cut intervals, {len(keep_intervals)} keep intervals")
            
            # Step 8: Remap captions to edited timeline
            if progress_callback:
                progress_callback({"step": "remapping_captions", "percent": 80})
            
            remapped_captions = remap_captions(captions, merged_cuts, words)
            remapped_captions = reindex_captions(remapped_captions)
            
            logger.info(f"Remapped {len(remapped_captions)} captions to edited timeline")
            
            # Step 9: Build analysis result
            if progress_callback:
                progress_callback({"step": "building_result", "percent": 90})
            
            analysis = AnalysisResult(
                media_info=media_info,
                filler_detections=filler_detections,
                words=words,
                full_transcript=full_transcript,
                captions=remapped_captions,
                cut_padding=self.cut_padding,
                approved_cut_intervals=merged_cuts,
                keep_intervals=keep_intervals,
                settings={
                    "removable_fillers": [ft.value for ft in self.removable_fillers],
                    "confidence_threshold": self.confidence_threshold,
                    "min_filler_duration": self.min_filler_duration,
                    "max_filler_duration": self.max_filler_duration,
                    "whisper_model": self.whisper_model
                }
            )
            
            # Cache the result
            self._last_analysis = analysis
            
            if progress_callback:
                progress_callback({"step": "complete", "percent": 100})
            
            logger.info(f"Analysis complete: {analysis}")
            return analysis
            
        finally:
            # Clean up temp audio file
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass
    
    def _detect_fillers(self, audio_path: str, progress_callback=None) -> List[FillerDetection]:
        """Detect fillers using UHM.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of FillerDetection objects
        """
        try:
            if self._uhm_model is None:
                self._uhm_model = UHMModel.from_cache()
            
            detector = FillerDetector(self._uhm_model)
            detections = detector.detect_from_file(
                audio_path,
                removable_types=self.removable_fillers,
                confidence_threshold=self.confidence_threshold,
                progress_callback=progress_callback,
            )
            return detections
            
        except AnalysisCancelled:
            raise
        except Exception as e:
            logger.error(f"Filler detection failed: {e}")
            raise AnalysisError(f"Filler detection failed: {e}")
    
    def _transcribe(self, audio_path: str) -> Tuple[str, List[Word]]:
        """Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (full_transcript, list_of_words)
        """
        try:
            if (self._whisper_transcriber is None or
                    self._whisper_transcriber.model_size != self.whisper_model):
                self._whisper_transcriber = WhisperTranscriber(self.whisper_model)
            result = self._whisper_transcriber.transcribe(audio_path)
            return get_full_transcript(result), convert_to_words(result)
        except AnalysisCancelled:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise AnalysisError(f"Transcription failed: {e}")
    
    def export_timeline(
        self,
        analysis: AnalysisResult,
        output_path: str,
        timeline_name: Optional[str] = None
    ) -> str:
        """Export FCPXML timeline from analysis.
        
        Args:
            analysis: AnalysisResult to export
            output_path: Path to save FCPXML
            timeline_name: Optional timeline name
            
        Returns:
            Path to the exported FCPXML file
        """
        try:
            return generate_fcpxml_from_analysis(
                analysis,
                output_path,
                timeline_name=timeline_name
            )
        except Exception as e:
            raise AnalysisError(f"Failed to export timeline: {e}")
    
    def export_srt(
        self,
        analysis: AnalysisResult,
        output_path: str
    ) -> str:
        """Export SRT subtitles from analysis.
        
        Args:
            analysis: AnalysisResult to export
            output_path: Path to save SRT
            
        Returns:
            Path to the exported SRT file
        """
        try:
            save_srt(analysis.captions, output_path)
            logger.info(f"SRT exported to {output_path}")
            return output_path
        except Exception as e:
            raise AnalysisError(f"Failed to export SRT: {e}")
    
    def export_json(
        self,
        analysis: AnalysisResult,
        output_path: str
    ) -> str:
        """Export JSON analysis from analysis.
        
        Args:
            analysis: AnalysisResult to export
            output_path: Path to save JSON
            
        Returns:
            Path to the exported JSON file
        """
        try:
            json_str = analysis.to_json()
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"JSON exported to {output_path}")
            return output_path
        except Exception as e:
            raise AnalysisError(f"Failed to export JSON: {e}")
    
    def export_all(
        self,
        analysis: AnalysisResult,
        output_dir: str,
        base_name: Optional[str] = None
    ) -> Dict[str, str]:
        """Export all outputs (FCPXML, SRT, JSON) to a directory.
        
        Args:
            analysis: AnalysisResult to export
            output_dir: Directory to save outputs
            base_name: Base filename (without extension)
            
        Returns:
            Dictionary with paths to exported files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(analysis.media_info.path))[0]
        
        results = {}
        
        # Export timeline
        timeline_path = os.path.join(output_dir, f"{base_name}.fcpxml")
        results["fcpxml"] = self.export_timeline(analysis, timeline_path)
        
        # Export SRT
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        results["srt"] = self.export_srt(analysis, srt_path)
        
        # Export JSON
        json_path = os.path.join(output_dir, f"{base_name}.json")
        results["json"] = self.export_json(analysis, json_path)
        
        logger.info(f"All exports complete: {results}")
        return results
    
    def update_detection_enabled(
        self,
        analysis: AnalysisResult,
        detection_index: int,
        enabled: bool
    ) -> AnalysisResult:
        """Update whether a detection is enabled for removal.
        
        Args:
            analysis: AnalysisResult to modify
            detection_index: Index of the detection to update
            enabled: Whether to enable or disable
            
        Returns:
            Updated AnalysisResult
        """
        if not 0 <= detection_index < len(analysis.filler_detections):
            raise IndexError(f"Detection index out of range: {detection_index}")
        analysis.filler_detections[detection_index].enabled = enabled
        
        # Recalculate cut intervals
        cut_intervals = []
        for det in analysis.filler_detections:
            if det.enabled:
                padded = det.pad(analysis.cut_padding)
                interval = padded.interval.clamp(0, analysis.media_info.duration)
                cut_intervals.append(interval)
        
        merged_cuts = snap_cut_intervals_to_frames(
            analysis.media_info, merge_overlapping(cut_intervals)
        )
        keep_intervals = invert_intervals(merged_cuts, analysis.media_info.duration)
        
        analysis.approved_cut_intervals = merged_cuts
        analysis.keep_intervals = keep_intervals
        
        # Always rebuild source-timeline captions from the immutable word
        # timestamps. Remapping already-edited captions compounds offsets and
        # cannot restore words when a cut is disabled.
        source_captions = generate_captions_from_words(analysis.words)
        analysis.captions = remap_captions(source_captions, merged_cuts, analysis.words)
        analysis.captions = reindex_captions(analysis.captions)
        
        return analysis
    
    def set_cut_padding(self, padding: float) -> None:
        """Set the cut padding.
        
        Args:
            padding: New padding value in seconds
        """
        if padding < 0:
            raise ValueError("Cut padding cannot be negative")
        self.cut_padding = padding

    def update_cut_padding(self, analysis: AnalysisResult, padding: float) -> AnalysisResult:
        """Apply a new padding value and deterministically rebuild edit state."""
        self.set_cut_padding(padding)
        analysis.cut_padding = padding
        if analysis.filler_detections:
            return self.update_detection_enabled(
                analysis, 0, analysis.filler_detections[0].enabled
            )

        analysis.approved_cut_intervals = []
        analysis.keep_intervals = invert_intervals([], analysis.media_info.duration)
        analysis.captions = reindex_captions(generate_captions_from_words(analysis.words))
        return analysis


def run_analysis_pipeline(
    media_path: str,
    output_dir: str,
    cut_padding: float = 0.05,
    removable_fillers: set = DEFAULT_REMOVABLE_FILLERS,
    whisper_model: str = "base",
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Run the complete analysis pipeline and export all outputs.
    
    This is a convenience function for running the entire pipeline
    in one call.
    
    Args:
        media_path: Path to the media file
        output_dir: Directory to save outputs
        cut_padding: Padding for cuts
        removable_fillers: Set of FillerType to mark as enabled
        whisper_model: Whisper model size
        progress_callback: Function to report progress
        
    Returns:
        Dictionary with analysis results and exported file paths
    """
    app = ResolveAutoCut(
        cut_padding=cut_padding,
        removable_fillers=removable_fillers,
        whisper_model=whisper_model
    )
    
    # Run analysis
    analysis = app.analyze(media_path, progress_callback)
    
    # Export all outputs
    exports = app.export_all(analysis, output_dir)
    
    return {
        "analysis": analysis,
        "exports": exports
    }


def create_sample_analysis(media_path: str) -> AnalysisResult:
    """Create a sample analysis for testing purposes.
    
    This creates a synthetic analysis with some fake detections
    for testing the pipeline without running ML models.
    
    Args:
        media_path: Path to use for media info
        
    Returns:
        AnalysisResult with sample data
    """
    from .media import inspect_media
    
    media_info = inspect_media(media_path)
    
    # Create some sample filler detections
    detections = [
        FillerDetection(
            start=5.0, end=5.5, filler_type=FillerType.UH,
            confidence=0.95, enabled=True
        ),
        FillerDetection(
            start=15.0, end=15.3, filler_type=FillerType.UM,
            confidence=0.85, enabled=True
        ),
        FillerDetection(
            start=20.0, end=20.8, filler_type=FillerType.HMM,
            confidence=0.75, enabled=True
        ),
        FillerDetection(
            start=30.0, end=30.2, filler_type=FillerType.UH,
            confidence=0.60, enabled=False  # Disabled
        ),
    ]
    
    # Create sample words
    words = [
        Word(text="Hello", start=0.0, end=0.5, index=0),
        Word(text="everyone", start=0.5, end=1.0, index=1),
        Word(text="today", start=1.0, end=1.5, index=2),
        Word(text="we", start=4.0, end=4.5, index=3),
        Word(text="are", start=4.5, end=4.8, index=4),
        Word(text="going", start=4.8, end=5.0, index=5),
        # Filler at 5.0-5.5
        Word(text="to", start=5.5, end=5.8, index=6),
        Word(text="discuss", start=5.8, end=6.5, index=7),
    ]
    
    # Create sample captions
    captions = [
        Caption(index=1, start=0.0, end=1.5, text="Hello everyone today"),
        Caption(index=2, start=4.0, end=5.0, text="We are going"),
        Caption(index=3, start=5.5, end=6.5, text="To discuss"),
    ]
    
    # Calculate intervals
    cut_padding = 0.05
    cut_intervals = []
    for det in detections:
        if det.enabled:
            padded = det.pad(cut_padding)
            interval = padded.interval.clamp(0, media_info.duration)
            cut_intervals.append(interval)
    
    merged_cuts = merge_overlapping(cut_intervals)
    keep_intervals = invert_intervals(merged_cuts, media_info.duration)
    
    # Remap captions
    remapped_captions = remap_captions(captions, merged_cuts)
    remapped_captions = reindex_captions(remapped_captions)
    
    return AnalysisResult(
        media_info=media_info,
        filler_detections=detections,
        words=words,
        full_transcript="Hello everyone today we are going to discuss",
        captions=remapped_captions,
        cut_padding=cut_padding,
        approved_cut_intervals=merged_cuts,
        keep_intervals=keep_intervals,
        settings={"test": True}
    )
