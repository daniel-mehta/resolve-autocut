"""Frame-aligned, ripple-delete FCPXML 1.9 export."""
import os
import math
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET
from xml.dom import minidom

from .models import AnalysisResult, Interval, MediaInfo
from .intervals import merge_overlapping, sort_intervals, invert_intervals

FCPXML_NAMESPACE = "http://www.apple.com/fcpxml"
FCPXML_VERSION = "1.9"

class TimelineError(Exception): pass


def get_frame_rate_fraction(fps: float) -> Fraction:
    common = {23.976: Fraction(24000,1001), 24: Fraction(24), 25: Fraction(25),
              29.97: Fraction(30000,1001), 30: Fraction(30), 50: Fraction(50),
              59.94: Fraction(60000,1001), 60: Fraction(60)}
    return common.get(round(fps, 3), Fraction(fps).limit_denominator(100000))

def fraction_to_fcpxml(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"

def seconds_to_frames(seconds: float, frame_rate: Fraction) -> int:
    return round(Fraction(str(seconds)) * frame_rate)

def frames_to_seconds(frames: int, frame_rate: Fraction) -> float:
    return float(Fraction(frames, 1) / frame_rate)

def _time(frames: int, rate: Fraction) -> str:
    value = Fraction(frames, 1) / rate
    return f"{value.numerator}/{value.denominator}s"

def _source_uri(path: str) -> str:
    return Path(path).resolve().as_uri()

def _frame_cut_ranges(media: MediaInfo, cuts: List[Interval], rate: Fraction) -> List[Tuple[int, int]]:
    if rate <= 0:
        raise TimelineError(f"Invalid frame rate: {rate}")
    if media.duration < 0:
        raise TimelineError(f"Invalid media duration: {media.duration}")

    normalized_cuts = merge_overlapping(sort_intervals(cuts))
    for cut in normalized_cuts:
        if cut.start < 0 or cut.end > media.duration + 1e-7:
            raise TimelineError(f"Cut interval outside media duration: {cut}")

    # The source may end partway through its nominal final frame. Keep that
    # frame rather than silently shortening the asset.
    total = math.ceil(Fraction(str(media.duration)) * rate)

    # Snap interior cuts inward. Only frames wholly inside a requested cut are
    # removed, which is the conservative choice for neighboring phonemes.
    # Cuts touching a source edge may consume the partial edge frame.
    frame_cuts: List[Tuple[int, int]] = []
    for cut in normalized_cuts:
        start_value = Fraction(str(cut.start)) * rate
        end_value = Fraction(str(cut.end)) * rate
        start = 0 if cut.start <= 1e-10 else math.ceil(start_value)
        end = total if cut.end >= media.duration - 1e-10 else math.floor(end_value)
        start, end = max(0, start), min(total, end)
        if end > start:
            frame_cuts.append((start, end))

    merged_cuts: List[Tuple[int, int]] = []
    for start, end in frame_cuts:
        if merged_cuts and start <= merged_cuts[-1][1]:
            merged_cuts[-1] = (merged_cuts[-1][0], max(end, merged_cuts[-1][1]))
        else:
            merged_cuts.append((start, end))

    return merged_cuts


def snap_cut_intervals_to_frames(
    media: MediaInfo,
    cuts: List[Interval],
    frame_rate: Optional[Fraction] = None,
) -> List[Interval]:
    """Return the exact conservative cuts that the FCPXML will apply."""
    rate = frame_rate or media.frame_rate or Fraction(24)
    if not isinstance(rate, Fraction):
        rate = get_frame_rate_fraction(float(rate))
    return [
        Interval(float(Fraction(start, 1) / rate),
                 min(float(Fraction(end, 1) / rate), media.duration))
        for start, end in _frame_cut_ranges(media, cuts, rate)
    ]


def _frame_intervals(media: MediaInfo, keeps: List[Interval], rate: Fraction) -> List[Tuple[int, int]]:
    normalized_keeps = merge_overlapping(sort_intervals(keeps))
    for keep in normalized_keeps:
        if keep.start < 0 or keep.end > media.duration + 1e-7:
            raise TimelineError(f"Keep interval outside media duration: {keep}")
    total = math.ceil(Fraction(str(media.duration)) * rate)
    merged_cuts = _frame_cut_ranges(
        media, invert_intervals(normalized_keeps, media.duration), rate
    )
    result: List[Tuple[int, int]] = []
    cursor = 0
    for start, end in merged_cuts:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        result.append((cursor, total))
    return result

def generate_fcpxml(media_info: MediaInfo, keep_intervals: List[Interval], output_path: str,
                    frame_rate: Optional[Fraction] = None,
                    timeline_name: str = "Resolve AutoCut Timeline") -> str:
    rate = frame_rate or media_info.frame_rate or Fraction(24)
    if not isinstance(rate, Fraction): rate = get_frame_rate_fraction(float(rate))
    clips = _frame_intervals(media_info, keep_intervals, rate)
    total_frames = math.ceil(Fraction(str(media_info.duration)) * rate)
    retained_frames = sum(end - start for start, end in clips)
    root = ET.Element("fcpxml", {"version": FCPXML_VERSION})
    resources = ET.SubElement(root, "resources")
    ET.SubElement(resources, "format", {"id":"r1", "name":f"ResolveAutoCut {media_info.width or 1920}x{media_info.height or 1080}",
        "frameDuration":_time(1,rate), "width":str(media_info.width or 1920), "height":str(media_info.height or 1080)})
    asset_attrs = {"id":"r2", "name":os.path.basename(media_info.path), "src":_source_uri(media_info.path),
        "start":"0s", "duration":_time(total_frames,rate)}
    if media_info.width or media_info.height or media_info.frame_rate:
        asset_attrs.update({"hasVideo":"1", "format":"r1"})
    if media_info.sample_rate:
        asset_attrs.update({"hasAudio":"1", "audioSources":"1", "audioChannels":str(media_info.channels or 1), "audioRate":str(media_info.sample_rate)})
    if "hasVideo" not in asset_attrs and "hasAudio" not in asset_attrs:
        raise TimelineError("Media has neither a video nor an audio stream")
    ET.SubElement(resources, "asset", asset_attrs)
    library = ET.SubElement(root,"library")
    event = ET.SubElement(library,"event",{"name":"Resolve AutoCut"})
    project = ET.SubElement(event,"project",{"name":timeline_name})
    sequence = ET.SubElement(project,"sequence",{"format":"r1", "duration":_time(retained_frames,rate), "tcStart":"0s", "tcFormat":"NDF"})
    spine = ET.SubElement(sequence,"spine")
    offset = 0
    for start, end in clips:
        duration = end - start
        # Offsets intentionally accumulate only retained duration: ripple delete.
        ET.SubElement(spine,"asset-clip",{"name":os.path.basename(media_info.path), "ref":"r2",
            "offset":_time(offset,rate), "start":_time(start,rate), "duration":_time(duration,rate)})
        offset += duration
    xml = minidom.parseString(ET.tostring(root,encoding="utf-8")).toprettyxml(indent="  ")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path,"w",encoding="utf8") as handle: handle.write(xml)
    return output_path

def generate_fcpxml_from_analysis(analysis: AnalysisResult, output_path: str, timeline_name: Optional[str] = None) -> str:
    return generate_fcpxml(analysis.media_info, analysis.keep_intervals, output_path,
        analysis.media_info.frame_rate, timeline_name or f"AutoCut - {os.path.basename(analysis.media_info.path)}")

def create_simple_fcpxml(media_path: str, cut_intervals: List[Interval], output_path: str, frame_rate: Optional[float] = None) -> str:
    from .media import inspect_media
    media = inspect_media(media_path)
    return generate_fcpxml(media, invert_intervals(cut_intervals, media.duration), output_path,
                           get_frame_rate_fraction(frame_rate) if frame_rate else media.frame_rate)

def calculate_timeline_duration(keep_intervals: List[Interval], media_duration: Optional[float] = None) -> float:
    return sum(interval.duration for interval in keep_intervals)

def get_clip_info(fcpxml_path: str):
    root = ET.parse(fcpxml_path).getroot()
    return [{"name":clip.get("name"),"offset":clip.get("offset"),"start":clip.get("start"),"duration":clip.get("duration"),"ref":clip.get("ref")}
            for clip in root.findall(".//asset-clip")]

def validate_fcpxml(fcpxml_path: str) -> Tuple[bool,List[str]]:
    try: root=ET.parse(fcpxml_path).getroot()
    except Exception as exc: return False,[str(exc)]
    errors=[]
    if root.tag != "fcpxml" or root.get("version") != FCPXML_VERSION: errors.append("Not an FCPXML 1.9 document")
    formats = {item.get("id"): item for item in root.findall("./resources/format")}
    asset_elements = root.findall("./resources/asset")
    assets={a.get("id"): a for a in asset_elements}
    if not formats: errors.append("Missing format resource")
    if not assets: errors.append("Missing asset resource")
    for asset in asset_elements:
        if not all(asset.get(key) for key in ("id", "name", "src", "duration")):
            errors.append("Asset missing required attributes")
        if asset.get("hasVideo") == "1" and asset.get("format") not in formats:
            errors.append(f"Unknown asset format {asset.get('format')}")
    sequence=root.find(".//sequence")
    if sequence is None: errors.append("Missing sequence")
    clips = root.findall(".//asset-clip")
    expected_offset = Fraction(0)
    retained = Fraction(0)
    for clip in clips:
        if clip.get("ref") not in assets: errors.append(f"Unknown asset ref {clip.get('ref')}")
        if not all(clip.get(key) for key in ("offset","start","duration")):
            errors.append("Clip missing timing")
            continue
        try:
            offset = _parse_time(clip.get("offset"))
            start = _parse_time(clip.get("start"))
            duration = _parse_time(clip.get("duration"))
            if duration <= 0: errors.append("Clip has non-positive duration")
            if offset != expected_offset:
                errors.append(f"Non-contiguous clip offset {clip.get('offset')}; expected {_fraction_time(expected_offset)}")
            asset = assets.get(clip.get("ref"))
            if asset is not None and start + duration > _parse_time(asset.get("duration")):
                errors.append("Clip exceeds asset duration")
            expected_offset = offset + duration
            retained += duration
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"Invalid clip timing: {exc}")
    if sequence is not None and sequence.get("duration"):
        try:
            if _parse_time(sequence.get("duration")) != retained:
                errors.append("Sequence duration does not equal retained clip duration")
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"Invalid sequence duration: {exc}")
    return not errors,errors


def _parse_time(value: str) -> Fraction:
    if not value or not value.endswith("s"):
        raise ValueError(f"Invalid FCPXML time {value!r}")
    return Fraction(value[:-1])


def _fraction_time(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}s"
