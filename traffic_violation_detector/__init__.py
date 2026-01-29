"""
Traffic Violation Detection Pipeline.

A complete computer vision pipeline for detecting traffic violations:
- Two-wheeler (motorcycle/scooter) detection
- Helmet violation detection
- License plate detection and OCR
- Violation logging to CSV

Usage:
    from traffic_violation_detector import TrafficViolationPipeline
    
    pipeline = TrafficViolationPipeline()
    result = pipeline.process_image("path/to/image.jpg")
    print(f"Violations found: {result.violations_found}")
"""

from .src.config import PipelineConfig, load_config
from .src.models import (
    BoundingBox,
    Detection,
    LicensePlateDetection,
    ProcessingResult,
    RiderDetection,
    TwoWheelerDetection,
    Violation,
    ViolationType,
)
from .src.pipeline import TrafficViolationPipeline

__version__ = "1.0.0"
__all__ = [
    "TrafficViolationPipeline",
    "PipelineConfig",
    "load_config",
    "Violation",
    "ViolationType",
    "ProcessingResult",
    "BoundingBox",
    "Detection",
    "TwoWheelerDetection",
    "RiderDetection",
    "LicensePlateDetection",
]
