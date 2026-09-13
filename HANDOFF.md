# Resolve AutoCut Engineering Handoff

## Audit baseline and scope

The independent release audit started from commit `de0b17b` (baseline
`598b331`). The source interview under `samples/` was read only. Model files
remain in the Hugging Face cache, and validation exports remain gitignored
under `outputs/`. No commit, push, deployment, or media render was performed.

## Real-media validation

The official cached `desert-ant-labs/uhm` `uhm-web-fp16.onnx` and MLX Whisper
0.4.3 `tiny` model completed the full 1,913.024-second interview on this host.
The final conservative run used the 0.75 confidence threshold and took 89.12
seconds end to end:

- 64 UHM model windows completed; the 1,499-frame output coverage advances by
  29.98 seconds, with an intentional 20 ms input-context overlap so no source
  time is skipped.
- 6 detections, all classed `um`, confidence 0.752793–0.848559.
- 5,362 real timestamped Whisper words and 702 edited captions.
- Six padded proposals span 4.98 seconds before frame alignment. The 16 fps
  export conservatively removes 73 complete frames (4.5625 seconds), leaving
  1,908.4615 seconds on the source clock. Its full-frame FCPXML sequence is
  exactly 3,817/2 seconds (1,908.5 seconds), including the partial final source
  frame.
- No detections were clustered within 100 ms of a chunk start, source word
  timestamps did not regress, captions were monotonic/non-overlapping, and
  cuts plus keeps covered the source duration exactly.
- The 16 fps FCPXML contains 7 contiguous retained clips. Its exact rational
  sequence duration equals the sum of clip durations, with no destination gaps.

Release-validation outputs:

- `outputs/full-interview-validation/<source-basename>.fcpxml`
- `outputs/full-interview-validation/<source-basename>.srt`
- `outputs/full-interview-validation/<source-basename>.json`

An exploratory 0.50-threshold full run found 65 candidates and proposed 40.26
seconds of cuts. Because upstream describes 0.50 as a recall/review setting and
0.75 as its precision preset for automatic cuts, the application default is
now 0.75. Every proposed cut still requires human review.

## Correctness fixes in this audit

- Closed the 20 ms blind spot between 30-second UHM inputs and the model's
  29.98-second output coverage; padded-tail predictions remain clamped to the
  real source duration.
- Made detection toggles rebuild captions from immutable source words. The old
  GUI compounded timestamp shifts and could not restore words when cuts were
  disabled.
- Assigned numeric Treeview row IDs so double-click toggling no longer attempts
  `int("I001")`.
- Added editable-padding recalculation, cooperative cancellation, worker-safe
  queued logging, duplicate-analysis guards, and safer file-selection state.
- Removed a redundant FFmpeg re-extraction before Whisper and reused the
  transcriber wrapper.
- Export now snaps interior cuts inward to whole frames, preserves a partial
  final source frame, and validates exact rational ripple offsets/durations.
- Caption and JSON edit state now use those same snapped cut boundaries; the
  prior export and subtitle clocks diverged by 0.4175 seconds on the 16 fps run.
- SRT output now renumbers cues sequentially, rounds milliseconds correctly,
  skips invalid zero-duration cues, and preserves Unicode text.
- Restored the documented top-level Python API imports.
- Removed the unused `pydub` dependency and refreshed `uv.lock`.

## Installation and tests

`UV_PROJECT_ENVIRONMENT=<temporary-environment> uv sync --all-extras --locked`
succeeded in a fresh Python 3.13.9 environment. Top-level API and GUI module
imports succeeded there. The full suite passed in that environment:

```text
<temporary-environment>/bin/python -m pytest -q
149 passed in 0.37s
```

Coverage includes all requested UHM boundary durations, 16 fps plus integer and
NTSC fractional rates, exact ripple invariants, special/Unicode media URIs,
reversible caption review, and SRT validity.

## Remaining manual validation

1. Import the full-interview FCPXML above into the target DaVinci Resolve Free
   version; confirm automatic relink, picture/audio sync, cut placement, and
   final timeline duration.
2. Run one GUI review cycle manually: select media, analyze, disable/re-enable a
   detection, change padding, export, and confirm the revised selection.

The Tk GUI process launched natively during this audit, but the automation
surface could not attach to the Python/Tk window. Actual visual interaction and
Resolve import therefore remain explicit human gates, not claimed validations.

## Licensing and release language

Resolve AutoCut-authored code is MIT. UHM is not MIT or open source; it uses the
Desert Ant Labs Source-Available License 1.0. Other dependencies retain their
own licenses. Media inference is local, but dependency/model downloads contact
package registries and Hugging Face. README claims were narrowed accordingly.
