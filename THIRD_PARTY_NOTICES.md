# Third-Party Notices

This document provides attribution and licensing information for third-party software and models used by Resolve AutoCut.

**Important: The MIT License applies only to the source code authored specifically for Resolve AutoCut. Third-party software and models retain their original licenses and copyrights.**

## Filler Detection: UHM by Desert Ant Labs

Resolve AutoCut uses UHM (Uh-and-Um Detector) for filler word detection.

- **Model**: `uhm-web-fp16.onnx`
- **Repository**: https://huggingface.co/desert-ant-labs/uhm
- **Homepage**: https://desertant.com/
- **License**: Desert Ant Labs Source-Available License 1.0
- **License URL**: https://license.desertant.com/1.0

**Attribution**: Filler detection powered by UHM by Desert Ant Labs.

The UHM model is downloaded at runtime from Hugging Face Hub using the official `huggingface_hub.hf_hub_download()` mechanism. The model weights are cached in the user's Hugging Face cache directory and are **not** committed to this repository. Users must accept the model's license terms when using this feature.

### Usage in Resolve AutoCut

```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="desert-ant-labs/uhm",
    filename="uhm-web-fp16.onnx"
)
```

## Transcription: MLX Whisper

Resolve AutoCut uses MLX-Whisper for local speech-to-text transcription optimized for Apple Silicon.

- **Repository**: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- **PyPI**: https://pypi.org/project/mlx-whisper/
- **License**: MIT License (mlx-whisper)
- **Dependencies**: MLX (Apple Silicon ML framework)

MLX and MLX-Whisper are designed specifically for Apple Silicon Macs and provide fast, local transcription without cloud APIs.

## FFmpeg / FFprobe

Resolve AutoCut uses FFmpeg for media inspection, audio extraction, and format conversion.

- **Homepage**: https://ffmpeg.org/
- **License**: LGPL / GPL (depending on configuration)
- **Pre-built binaries**: Homebrew (`brew install ffmpeg`)

FFmpeg is called via subprocess and must be installed separately on the user's system.

## ONNX Runtime

Resolve AutoCut uses ONNX Runtime to run the UHM ONNX model.

- **Repository**: https://github.com/microsoft/onnxruntime
- **License**: MIT License
- **PyPI**: https://pypi.org/project/onnxruntime/

## Hugging Face Hub

Resolve AutoCut uses the Hugging Face Hub Python library to download UHM model weights.

- **Repository**: https://github.com/huggingface/huggingface_hub
- **License**: Apache License 2.0
- **PyPI**: https://pypi.org/project/huggingface-hub/

## NumPy

- **Homepage**: https://www.numpy.org/
- **License**: BSD 3-Clause
- **PyPI**: https://pypi.org/project/numpy/

---

## Summary of Licensing

| Component | License | Source |
|-----------|---------|--------|
| Resolve AutoCut (this project) | MIT | This repository |
| UHM Model | Desert Ant Labs Source-Available 1.0 | Desert Ant Labs |
| MLX-Whisper | MIT | ml-explore |
| MLX | MIT | ml-explore |
| ONNX Runtime | MIT | Microsoft |
| Hugging Face Hub | Apache 2.0 | Hugging Face |
| NumPy | BSD 3-Clause | NumPy Project |
| FFmpeg | LGPL/GPL | FFmpeg Project |

**Note**: Users must ensure they have the right to use all third-party components in their jurisdiction and for their intended use case.
