# Resolve AutoCut Engineering Handoff

## Baseline and scope

Baseline was `598b331`. Source interview footage under `samples/` was not
modified. Model files remain in the normal Hugging Face cache and generated
exports are gitignored under `outputs/`.

## What was actually verified

- FFprobe inspected the supplied 31:53 sample (1920x1080 H.264, 16 fps,
  16 kHz mono AAC).
- Official UHM `uhm-web-fp16.onnx` downloaded and loaded. ONNX inspection:
  input `audio float32 [1,480000]`; output `probs float32 [1,1499,6]`.
  UHM documentation confirms six softmax classes: not_filler, uh, um, hmm,
  and, other.
- UHM ran on real 30- and 60-second excerpts. The first 30 seconds produced
  an `um` at 7.12–7.52 s at 0.656 confidence.
- MLX Whisper 0.4.3 tiny downloaded/ran on real audio using its supported API.
  The first 30 seconds produced 77 real timestamped words.
- Complete pipeline ran on a lossless stream-copy 60-second excerpt of the
  local interview: 3 `um` detections (0.510–0.656), 132 words, 20 captions,
  3 cuts, 1.54 s removed, 58.492 s float retained duration.
- FCPXML/XML, SRT, and JSON exports were generated and parsed. The FCPXML has
  one asset and ripple offsets accumulated solely from retained frames.

## Important fixes

- Replaced invalid UHM output assumptions and catastrophic 20 ms / 30-second
  overlapping inference loop with contiguous 30-second model windows.
- Replaced nonexistent MLX Whisper `load_model`/`load_processor` and invalid
  `return_word_timestamps` call with the installed API. Timestamp fabrication
  was removed.
- Fixed empty `NamedTemporaryFile` audio being incorrectly treated as cached
  extraction output.
- Rebuilt FCPXML as FCPXML 1.9 resources + `asset-clip` spine entries using
  rational, frame-snapped source/destination durations and ripple semantics.
- Caption remapping now removes timestamped words whose audio is cut and emits
  valid blank-separated SRT cues.

## Tests

`.venv/bin/python -m pytest` passes **111 tests** after the final correction.
Tests include explicit ripple offsets and cut-word caption removal.

## Remaining manual validation

Import `outputs/real-sample/interview-60s-autocut.fcpxml` in the target
DaVinci Resolve Free version. Confirm media relinks to `/tmp/resolve-autocut-60s.mp4`
or regenerate from the desired source on that machine. The generated `/tmp`
excerpt is intentionally not repository content. No Resolve GUI automation
was attempted.

## Known limitation

The complete 31:53 source pipeline was started but this session host ended the
single long command while UHM was running; it must not be described as a full
source completion. The exact full pipeline has succeeded on the 60-second
stream-copy excerpt.
