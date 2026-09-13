"""Regression coverage for defects found during the v0.1.0 adversarial audit."""

from fractions import Fraction
import sys
import types
from xml.etree import ElementTree as ET
import queue
import threading

import numpy as np
import pytest

import resolve_autocut
from resolve_autocut.app import ResolveAutoCut
from resolve_autocut.captions import generate_captions_from_words, generate_srt, remap_captions
from resolve_autocut.models import (
    AnalysisResult, Caption, FillerDetection, FillerType, Interval, MediaInfo, Word,
)
from resolve_autocut.gui import (
    ABOUT_TEXT, DESERT_ANT_URL, AnalysisThread, ResolveAutoCutGUI,
    _open_desert_ant_site,
)
from resolve_autocut.media import FFmpegNotFoundError, inspect_media
from resolve_autocut.transcription import (
    GUI_WHISPER_MODEL_SIZES, MLX_WHISPER_MODEL_REPOSITORIES,
    TranscriptionError, WhisperTranscriber, convert_to_words,
    resolve_whisper_model_repo,
)
from resolve_autocut.timeline import (
    generate_fcpxml, get_clip_info, snap_cut_intervals_to_frames, validate_fcpxml,
)
from resolve_autocut.uhm import (
    FillerDetector, UHM_OUTPUT_FRAMES, UHM_STRIDE,
    UHM_EXPECTED_SAMPLE_RATE, chunk_audio_for_uhm,
)


def test_documented_package_api_is_importable():
    assert resolve_autocut.ResolveAutoCut is ResolveAutoCut
    assert callable(resolve_autocut.run_analysis_pipeline)


def test_default_confidence_uses_upstream_precision_preset():
    assert ResolveAutoCut().confidence_threshold == 0.75


@pytest.mark.parametrize("kwargs", [
    {"cut_padding": -0.01},
    {"confidence_threshold": 1.01},
    {"min_filler_duration": 2.0, "max_filler_duration": 1.0},
    {"whisper_model": ""},
])
def test_invalid_analysis_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        ResolveAutoCut(**kwargs)


@pytest.mark.parametrize(("model_size", "repository"), [
    ("tiny", "mlx-community/whisper-tiny-mlx"),
    ("base", "mlx-community/whisper-base-mlx"),
    ("small", "mlx-community/whisper-small-mlx"),
    ("medium", "mlx-community/whisper-medium-mlx"),
])
def test_supported_whisper_sizes_resolve_to_mlx_repositories(model_size, repository):
    assert resolve_whisper_model_repo(model_size) == repository
    assert WhisperTranscriber(model_size).model_repo == repository


def test_gui_whisper_choices_are_all_supported_mlx_model_sizes():
    assert GUI_WHISPER_MODEL_SIZES == tuple(MLX_WHISPER_MODEL_REPOSITORIES)
    assert all(resolve_whisper_model_repo(choice).endswith("-mlx")
               for choice in GUI_WHISPER_MODEL_SIZES)


def test_about_surface_has_version_and_separate_uhm_attribution(monkeypatch):
    opened = []
    monkeypatch.setattr("resolve_autocut.gui.webbrowser.open", lambda url, new: opened.append((url, new)))

    assert "Resolve AutoCut v0.1.0" in ABOUT_TEXT
    assert "Filler detection powered by UHM by Desert Ant Labs." in ABOUT_TEXT
    assert "Resolve AutoCut-authored source code: MIT License" in ABOUT_TEXT
    assert "UHM: Desert Ant Labs Source-Available License 1.0" in ABOUT_TEXT
    _open_desert_ant_site()
    assert opened == [(DESERT_ANT_URL, 2)]


def test_explicit_whisper_repository_id_is_preserved():
    repository = "example-org/private-mlx-whisper"
    assert resolve_whisper_model_repo(repository) == repository
    assert WhisperTranscriber(repository).model_repo == repository


@pytest.mark.parametrize("model_size", ["", "base.en", "whisper-base", "owner/repo/extra"])
def test_invalid_whisper_model_name_is_rejected(model_size):
    with pytest.raises(ValueError, match="Unknown Whisper model"):
        resolve_whisper_model_repo(model_size)


def test_analysis_rejects_unknown_whisper_model_before_running_models():
    with pytest.raises(ValueError, match="Unknown Whisper model"):
        ResolveAutoCut(whisper_model="not-a-whisper-model")


def test_explicit_repository_authentication_failure_is_preserved(monkeypatch):
    def reject_private_repo(*args, **kwargs):
        raise RuntimeError("401 Client Error: Unauthorized for private repository")

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=reject_private_repo))
    with pytest.raises(TranscriptionError, match="401 Client Error: Unauthorized") as exc_info:
        WhisperTranscriber("example-org/private-mlx-whisper").transcribe("audio.wav")
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def _analysis_with_one_filler() -> AnalysisResult:
    words = [
        Word("hello", 0.0, 0.4, 0),
        Word("um", 0.4, 0.7, 1),
        Word("there", 0.7, 1.1, 2),
    ]
    detection = FillerDetection(0.4, 0.7, FillerType.UM, 0.9, True)
    cuts = [detection.interval]
    source = generate_captions_from_words(words)
    return AnalysisResult(
        media_info=MediaInfo("/tmp/interview.mp4", 1.1, Fraction(30)),
        filler_detections=[detection],
        words=words,
        captions=remap_captions(source, cuts, words),
        cut_padding=0.0,
        approved_cut_intervals=cuts,
        keep_intervals=[Interval(0.0, 0.4), Interval(0.7, 1.1)],
    )


def test_detection_toggle_rebuilds_captions_and_is_reversible():
    app = ResolveAutoCut(cut_padding=0.0)
    analysis = _analysis_with_one_filler()

    app.update_detection_enabled(analysis, 0, False)
    assert [caption.text.lower() for caption in analysis.captions] == ["hello um there."]
    assert [(caption.start, caption.end) for caption in analysis.captions] == [(0.0, 1.1)]

    app.update_detection_enabled(analysis, 0, True)
    first_enabled = [caption.to_dict() for caption in analysis.captions]
    app.update_detection_enabled(analysis, 0, True)
    assert [caption.to_dict() for caption in analysis.captions] == first_enabled
    assert [caption.text.lower() for caption in analysis.captions] == ["hello", "there"]


def test_multiple_cuts_reconstruct_caption_from_kept_word_runs():
    words = [
        Word("one", 0.0, 0.3, 0), Word("um", 0.3, 0.5, 1),
        Word("two", 0.5, 0.8, 2), Word("uh", 0.8, 1.0, 3),
        Word("three!", 1.0, 1.4, 4),
    ]
    source = [Caption(1, 0.0, 1.4, "one um two uh three!", list(range(5)))]
    result = remap_captions(source, [Interval(0.3, 0.5), Interval(0.8, 1.0)], words)
    assert [caption.text for caption in result] == ["One", "Two", "Three!"]
    assert [caption.start for caption in result] == pytest.approx([0.0, 0.3, 0.6])
    assert [caption.end for caption in result] == pytest.approx([0.3, 0.6, 1.0])


def test_partially_cut_ordinary_word_is_preserved_and_compressed():
    words = [Word("everything", 0.0, 1.0, 0)]
    source = [Caption(1, 0.0, 1.0, "everything", [0])]
    result = remap_captions(source, [Interval(0.2, 0.7)], words)
    assert [caption.text for caption in result] == ["Everything"]
    assert result[0].start == 0.0
    assert result[0].end == pytest.approx(0.5)


@pytest.mark.parametrize("duration", [29.99, 30.00, 30.01, 59.99, 60.00, 60.01, 1913.024])
def test_uhm_output_coverage_has_no_chunk_boundary_gap(duration):
    sample_count = round(duration * UHM_EXPECTED_SAMPLE_RATE)
    audio = np.zeros(sample_count, dtype=np.float32)
    chunks = chunk_audio_for_uhm(audio, UHM_EXPECTED_SAMPLE_RATE)
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == pytest.approx(duration, abs=1 / UHM_EXPECTED_SAMPLE_RATE)

    output_end = 0.0
    for start, end, chunk in chunks:
        assert start == pytest.approx(output_end)
        output_end = start + min(len(chunk) / UHM_EXPECTED_SAMPLE_RATE,
                                 UHM_OUTPUT_FRAMES * UHM_STRIDE)
    assert output_end == pytest.approx(duration, abs=1 / UHM_EXPECTED_SAMPLE_RATE)


def test_uhm_tail_is_clamped_and_boundary_filler_is_merged():
    class AllUmModel:
        def predict(self, chunk):
            probabilities = np.zeros((UHM_OUTPUT_FRAMES, 6), dtype=np.float32)
            probabilities[:, 2] = 0.9
            return probabilities

    duration = 30.01
    audio = np.zeros(round(duration * UHM_EXPECTED_SAMPLE_RATE), dtype=np.float32)
    detections = FillerDetector(AllUmModel()).detect_from_array(
        audio, UHM_EXPECTED_SAMPLE_RATE, confidence_threshold=0.5
    )
    assert len(detections) == 1
    assert detections[0].start == 0.0
    assert detections[0].end == pytest.approx(duration)


@pytest.mark.parametrize("rate", [
    Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001),
    Fraction(24), Fraction(25), Fraction(30), Fraction(50), Fraction(60), Fraction(16),
])
def test_fcpxml_offsets_are_exactly_contiguous_at_supported_rates(tmp_path, rate):
    media = MediaInfo("/tmp/source.mp4", 12.037, rate, sample_rate=48000, channels=2)
    output = tmp_path / "timeline.fcpxml"
    generate_fcpxml(
        media,
        [Interval(0, 1.013), Interval(1.087, 7.219), Interval(7.291, media.duration)],
        str(output),
    )
    valid, errors = validate_fcpxml(str(output))
    assert valid, errors
    clips = get_clip_info(str(output))
    expected = Fraction(0)
    retained = Fraction(0)
    for clip in clips:
        offset = Fraction(clip["offset"][:-1])
        duration = Fraction(clip["duration"][:-1])
        assert offset == expected
        expected += duration
        retained += duration
    sequence = ET.parse(output).getroot().find(".//sequence")
    assert Fraction(sequence.get("duration")[:-1]) == retained


def test_fcpxml_conservatively_snaps_interior_cut_and_keeps_partial_final_frame(tmp_path):
    media = MediaInfo("/tmp/source.mp4", 2.02, Fraction(24))
    output = tmp_path / "timeline.fcpxml"
    generate_fcpxml(media, [Interval(0, 1.01), Interval(1.09, 2.02)], str(output))
    clips = get_clip_info(str(output))
    assert [(clip["start"], clip["duration"]) for clip in clips] == [
        ("0/1s", "25/24s"), ("13/12s", "23/24s")
    ]
    asset = ET.parse(output).getroot().find("./resources/asset")
    assert asset.get("duration") == "49/24s"


def test_caption_cut_state_uses_same_frame_boundaries_as_fcpxml():
    media = MediaInfo("/tmp/source.mp4", 2.02, Fraction(24))
    cuts = snap_cut_intervals_to_frames(media, [Interval(1.01, 1.09)])
    assert cuts == [Interval(25 / 24, 26 / 24)]

    analysis = AnalysisResult(
        media_info=media,
        filler_detections=[FillerDetection(1.01, 1.09, FillerType.UM, 0.9)],
        words=[Word("before", 0.0, 1.0, 0), Word("after", 1.1, 2.0, 1)],
        cut_padding=0.0,
    )
    ResolveAutoCut(cut_padding=0.0).update_detection_enabled(analysis, 0, True)
    assert analysis.approved_cut_intervals == cuts
    assert analysis.total_removed_duration == pytest.approx(1 / 24)
    assert analysis.captions[-1].end == pytest.approx(2.0 - 1 / 24)


def test_fcpxml_uri_encodes_special_and_unicode_path(tmp_path):
    media_path = tmp_path / "Marta's interview (final) – 東京 & notes.mp4"
    output = tmp_path / "timeline.fcpxml"
    generate_fcpxml(MediaInfo(str(media_path), 1.0, Fraction(25)), [Interval(0, 1)], str(output))
    asset = ET.parse(output).getroot().find("./resources/asset")
    assert asset.get("src") == media_path.resolve().as_uri()
    assert "%20" in asset.get("src") and "%26" in asset.get("src")


def test_audio_only_fcpxml_does_not_claim_a_video_stream(tmp_path):
    output = tmp_path / "audio.fcpxml"
    media = MediaInfo("/tmp/interview audio.wav", 3.0, sample_rate=16000, channels=1)
    generate_fcpxml(media, [Interval(0, 3)], str(output))
    asset = ET.parse(output).getroot().find("./resources/asset")
    assert asset.get("hasAudio") == "1"
    assert asset.get("hasVideo") is None
    assert asset.get("format") is None
    assert validate_fcpxml(str(output)) == (True, [])


def test_fcpxml_validator_rejects_destination_gap(tmp_path):
    output = tmp_path / "timeline.fcpxml"
    media = MediaInfo("/tmp/source.mp4", 3.0, Fraction(24))
    generate_fcpxml(media, [Interval(0, 1), Interval(2, 3)], str(output))
    tree = ET.parse(output)
    tree.getroot().findall(".//asset-clip")[1].set("offset", "2/1s")
    tree.write(output, encoding="utf-8")
    valid, errors = validate_fcpxml(str(output))
    assert not valid
    assert any("Non-contiguous" in error for error in errors)


def test_srt_reindexes_skips_zero_duration_and_preserves_unicode():
    captions = [
        Caption(8, 0.0, 0.0, "discard"),
        Caption(12, 59.9996, 61.0, "Café — 東京!"),
    ]
    assert generate_srt(captions) == (
        "1\n00:01:00,000 --> 00:01:01,000\nCafé — 東京!\n\n"
    )


def test_transcription_rejects_missing_real_word_timestamps():
    with pytest.raises(TranscriptionError, match="no timestamped words"):
        convert_to_words({"text": "untimed transcript", "segments": []})


def test_analysis_thread_cancellation_is_reported_without_error():
    entered = threading.Event()
    release = threading.Event()

    class FakeApp:
        def analyze(self, path, progress_callback):
            entered.set()
            assert release.wait(2)
            progress_callback({"step": "next", "percent": 20})

    worker = AnalysisThread(FakeApp(), "/tmp/source.mp4", queue.Queue())
    worker.start()
    assert entered.wait(2)
    worker.cancel()
    release.set()
    worker.join(2)
    assert worker.cancelled
    assert worker.error is None


def test_gui_detection_rows_use_numeric_ids():
    inserted = []

    class FakeTree:
        def get_children(self): return []
        def insert(self, *args, **kwargs): inserted.append(kwargs["iid"])
        def tag_configure(self, *args, **kwargs): pass

    gui = ResolveAutoCutGUI.__new__(ResolveAutoCutGUI)
    gui.analysis = _analysis_with_one_filler()
    gui.detections_tree = FakeTree()
    gui._update_detections_table()
    assert inserted == ["0"]


def test_missing_ffmpeg_has_clear_error(monkeypatch):
    monkeypatch.setattr("resolve_autocut.media.check_ffmpeg", lambda: False)
    with pytest.raises(FFmpegNotFoundError, match="brew install ffmpeg"):
        inspect_media(__file__)
