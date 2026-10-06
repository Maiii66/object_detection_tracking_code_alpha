"""Configuration package for the detection + tracking pipeline.

Re-exports the public config API so callers can use::

    from config import AppConfig, build_parser

instead of the fully-qualified ``from config.config import AppConfig``.
"""

from .config import (
    CHI2INV95,
    DEFAULT_THREADS,
    PROJECT_ROOT,
    AppConfig,
    DetectorConfig,
    OutputConfig,
    TrackerConfig,
    VisualizationConfig,
    build_parser,
    detect_device,
    resolve_threads,
    setup_logging,
)

__all__ = [
    "AppConfig",
    "CHI2INV95",
    "DEFAULT_THREADS",
    "DetectorConfig",
    "OutputConfig",
    "PROJECT_ROOT",
    "TrackerConfig",
    "VisualizationConfig",
    "build_parser",
    "detect_device",
    "resolve_threads",
    "setup_logging",
]
