"""
Resolve AutoCut - AI-assisted video cleanup for DaVinci Resolve.
Local, open-source, Apple Silicon optimized.
"""

__version__ = "0.1.0"
__author__ = "Resolve AutoCut Team"

# Make the git revision available if possible
try:
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=__file__.rsplit("/", 3)[0]
    )
    if result.returncode == 0:
        __git_rev__ = result.stdout.strip()
    else:
        __git_rev__ = "unknown"
except Exception:
    __git_rev__ = "unknown"
