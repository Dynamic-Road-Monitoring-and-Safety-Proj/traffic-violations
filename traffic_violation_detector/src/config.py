"""
Configuration loader for the traffic violation detection pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelPaths:
    """Paths to all models used in the pipeline."""
    vehicle_detector: str
    helmet_detector: str
    plate_detector: str
    plate_ocr: str


@dataclass
class DetectionConfig:
    """Detection-related configuration."""
    two_wheeler_classes: list[int]
    helmet_classes: dict[str, int]
    thresholds: dict[str, float]
    max_riders: int = 2  # Maximum allowed riders on a two-wheeler


@dataclass
class ProcessingConfig:
    """Processing-related configuration."""
    frame_size: tuple[int, int]
    batch_size: int
    roi_expansion: float
    nms_iou: float


@dataclass
class OutputConfig:
    """Output-related configuration."""
    output_dir: str
    violations_csv: str
    save_images: bool
    images_dir: str


@dataclass
class LoggingConfig:
    """Logging-related configuration."""
    level: str
    log_file: str
    format: str


@dataclass
class AWSConfig:
    """AWS deployment configuration."""
    s3_bucket: str
    sqs_queue: str
    enable_s3_upload: bool


@dataclass
class PipelineConfig:
    """Main configuration container for the entire pipeline."""
    models: ModelPaths
    detection: DetectionConfig
    processing: ProcessingConfig
    output: OutputConfig
    logging: LoggingConfig
    aws: AWSConfig
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get absolute path to a model file."""
        model_rel_path = getattr(self.models, model_name)
        return self.project_root / model_rel_path
    
    def get_output_dir(self) -> Path:
        """Get absolute path to output directory."""
        return self.project_root / self.output.output_dir
    
    def get_violations_csv_path(self) -> Path:
        """Get absolute path to violations CSV file."""
        return self.get_output_dir() / self.output.violations_csv
    
    def get_images_dir(self) -> Path:
        """Get absolute path to violation images directory."""
        return self.get_output_dir() / self.output.images_dir
    
    def get_log_file_path(self) -> Path:
        """Get absolute path to log file."""
        return self.project_root / self.logging.log_file


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.yaml.
        
    Returns:
        PipelineConfig object with all configuration loaded.
    """
    if config_path is None:
        # __file__ is src/config.py, parent is src/, parent.parent is traffic_violation_detector/
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)
    
    # Parse sub-configurations
    models = ModelPaths(**raw_config["models"])
    
    detection = DetectionConfig(
        two_wheeler_classes=raw_config["detection"]["two_wheeler_classes"],
        helmet_classes=raw_config["detection"]["helmet_classes"],
        thresholds=raw_config["detection"]["thresholds"],
    )
    
    processing = ProcessingConfig(
        frame_size=tuple(raw_config["processing"]["frame_size"]),
        batch_size=raw_config["processing"]["batch_size"],
        roi_expansion=raw_config["processing"]["roi_expansion"],
        nms_iou=raw_config["processing"]["nms_iou"],
    )
    
    output = OutputConfig(**raw_config["output"])
    
    logging_config = LoggingConfig(**raw_config["logging"])
    
    aws = AWSConfig(**raw_config["aws"])
    
    return PipelineConfig(
        models=models,
        detection=detection,
        processing=processing,
        output=output,
        logging=logging_config,
        aws=aws,
        project_root=config_path.parent.parent,
    )
