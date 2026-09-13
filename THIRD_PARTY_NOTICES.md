# Third-Party Notices

The MIT license in this repository applies only to code authored for Resolve AutoCut. Third-party software, models, weights, and services remain subject to their own licenses and terms.

## UHM filler detector — Desert Ant Labs

Resolve AutoCut downloads and runs the `uhm-web-fp16.onnx` filler-detection model from [`desert-ant-labs/uhm`](https://huggingface.co/desert-ant-labs/uhm).

- Homepage: [Desert Ant Labs](https://desertant.com/)
- License: [Desert Ant Labs Source-Available License 1.0](https://license.desertant.com/1.0)

Attribution: Filler detection powered by UHM by Desert Ant Labs.

UHM is **not** MIT licensed, and Resolve AutoCut does not describe it as open source. Under the Desert Ant Labs Source-Available License 1.0, UHM use is free below **100,000 monthly active devices per platform, per model**; use above that threshold requires a commercial license. Review the authoritative [upstream license](https://license.desertant.com/1.0) for the complete terms and to determine whether they permit your intended use.

The model is downloaded at runtime through `huggingface_hub` and cached locally; its weights are not included in this repository.

## MLX Whisper and MLX

Resolve AutoCut uses [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for local transcription on Apple Silicon.

- MLX Whisper package: [PyPI](https://pypi.org/project/mlx-whisper/)
- MLX Whisper license: MIT
- MLX license: MIT

## FFmpeg and FFprobe

Resolve AutoCut calls [FFmpeg](https://ffmpeg.org/) and FFprobe for media inspection and audio extraction. They must be installed separately. Their license is LGPL or GPL depending on the build and configuration you install.

## Other direct dependencies

| Component | License | Reference |
| --- | --- | --- |
| ONNX Runtime | MIT | [Microsoft](https://github.com/microsoft/onnxruntime) |
| Hugging Face Hub | Apache-2.0 | [Hugging Face](https://github.com/huggingface/huggingface_hub) |
| NumPy | BSD 3-Clause | [NumPy](https://numpy.org/) |
| SciPy | BSD 3-Clause | [SciPy](https://scipy.org/) |
| SoundFile | BSD 3-Clause | [SoundFile](https://pypi.org/project/soundfile/) |

Dependency versions are recorded in `pyproject.toml` and `uv.lock`. This notice is an attribution summary, not legal advice; review the applicable upstream terms before distribution or commercial use.
