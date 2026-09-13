"""
Resolve AutoCut - local video cleanup for DaVinci Resolve workflows.
"""

__version__ = "0.1.0"
__author__ = "Resolve AutoCut Team"

from .app import AnalysisCancelled, AnalysisError, ResolveAutoCut, run_analysis_pipeline

__all__ = [
    "AnalysisCancelled",
    "AnalysisError",
    "ResolveAutoCut",
    "run_analysis_pipeline",
    "__version__",
]
