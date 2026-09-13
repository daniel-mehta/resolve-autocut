# Contributing to Resolve AutoCut

Thank you for helping improve this early v0.1.0 project.

## Bug reports and feature requests

Please use [GitHub Issues](https://github.com/daniel-mehta/resolve-autocut/issues) for bugs, unexpected edits, FCPXML import problems, and feature requests. A useful report includes the Resolve AutoCut version or commit, macOS version, Python version, DaVinci Resolve version (if relevant), media properties, steps to reproduce, expected behavior, actual behavior, and relevant logs or error messages.

Do not attach private interviews, sensitive recordings, credentials, or full local paths to a public issue. A short synthetic clip or a minimal reproducible description is usually enough.

## Development setup

Resolve AutoCut currently targets macOS on Apple Silicon with Python 3.12 or 3.13 and FFmpeg/FFprobe installed. From a clone:

```bash
brew install ffmpeg
uv sync --all-extras --locked
uv run python -m pytest
```

For a focused test, use a path such as:

```bash
uv run python -m pytest tests/test_timeline.py -v
```

## Pull requests

Keep pull requests focused, explain the user-visible change, and include tests when behavior changes. Do not commit model caches, generated exports, recordings, transcripts, or other private media. Before opening a pull request, run the relevant tests and `git diff --check`.

Please open an issue first for substantial design changes so the scope can be discussed.
