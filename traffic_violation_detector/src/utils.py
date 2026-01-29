"""
Utility functions for the traffic violation detection pipeline.
"""

import csv
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import BoundingBox, Violation


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Set up logging for the pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        log_format: Log message format.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("traffic_violation_detector")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def generate_violation_id() -> str:
    """Generate a unique violation ID."""
    return f"VIO-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"


def ensure_directories(paths: list[Path]) -> None:
    """Ensure all directories in the list exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


class ViolationCSVWriter:
    """Handles writing violations to CSV file."""
    
    CSV_HEADERS = [
        "violation_id",
        "timestamp",
        "violation_type",
        "license_plate",
        "plate_confidence",
        "vehicle_bbox",
        "image_path",
        "source_frame",
    ]
    
    def __init__(self, csv_path: Path):
        """
        Initialize CSV writer.
        
        Args:
            csv_path: Path to CSV file.
        """
        self.csv_path = csv_path
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                writer.writeheader()
    
    def write_violation(self, violation: Violation) -> None:
        """
        Append a violation record to the CSV file.
        
        Args:
            violation: Violation object to write.
        """
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writerow(violation.to_csv_row())
    
    def write_violations(self, violations: list[Violation]) -> None:
        """
        Append multiple violation records to the CSV file.
        
        Args:
            violations: List of Violation objects to write.
        """
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            for violation in violations:
                writer.writerow(violation.to_csv_row())


def draw_bbox(
    image: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw a bounding box on an image.
    
    Args:
        image: Image to draw on (BGR format).
        bbox: Bounding box to draw.
        color: BGR color tuple.
        thickness: Line thickness.
        label: Optional label text.
        font_scale: Font scale for label.
        
    Returns:
        Image with bounding box drawn.
    """
    img = image.copy()
    cv2.rectangle(img, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
    
    if label:
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            img,
            (bbox.x1, bbox.y1 - text_height - baseline - 5),
            (bbox.x1 + text_width, bbox.y1),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (bbox.x1, bbox.y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
        )
    
    return img


def annotate_violation(
    image: np.ndarray,
    vehicle_bbox: BoundingBox,
    plate_bbox: Optional[BoundingBox],
    plate_text: str,
    violation_type: str = "NO HELMET",
) -> np.ndarray:
    """
    Annotate an image with violation information.
    
    Args:
        image: Image to annotate (BGR format).
        vehicle_bbox: Bounding box of the vehicle.
        plate_bbox: Bounding box of the license plate (if detected).
        plate_text: Recognized license plate text.
        violation_type: Type of violation.
        
    Returns:
        Annotated image.
    """
    img = image.copy()
    
    # Draw vehicle bbox in red (violation)
    img = draw_bbox(
        img,
        vehicle_bbox,
        color=(0, 0, 255),
        thickness=3,
        label=f"VIOLATION: {violation_type}",
    )
    
    # Draw plate bbox in yellow
    if plate_bbox:
        img = draw_bbox(
            img,
            plate_bbox,
            color=(0, 255, 255),
            thickness=2,
            label=f"PLATE: {plate_text}",
        )
    
    return img


def save_violation_image(
    image: np.ndarray,
    output_dir: Path,
    violation_id: str,
) -> Path:
    """
    Save a violation image to disk.
    
    Args:
        image: Image to save (BGR format).
        output_dir: Directory to save image.
        violation_id: Violation ID for filename.
        
    Returns:
        Path to saved image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{violation_id}.jpg"
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), image)
    return filepath


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        Image in BGR format.
        
    Raises:
        FileNotFoundError: If image doesn't exist.
        ValueError: If image cannot be decoded.
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    
    return image


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    keep_aspect_ratio: bool = True,
) -> np.ndarray:
    """
    Resize an image to target size.
    
    Args:
        image: Image to resize.
        target_size: Target (width, height).
        keep_aspect_ratio: Whether to maintain aspect ratio with letterboxing.
        
    Returns:
        Resized image.
    """
    target_w, target_h = target_size
    
    if not keep_aspect_ratio:
        return cv2.resize(image, (target_w, target_h))
    
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create letterboxed image
    result = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First bounding box.
        box2: Second bounding box.
        
    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    union = box1.area + box2.area - intersection
    
    return intersection / union if union > 0 else 0.0


def is_bbox_inside(inner: BoundingBox, outer: BoundingBox, threshold: float = 0.5) -> bool:
    """
    Check if one bounding box is mostly inside another.
    
    Args:
        inner: The bounding box that should be inside.
        outer: The bounding box that should contain the inner one.
        threshold: Minimum fraction of inner box that must be inside outer.
        
    Returns:
        True if inner is mostly inside outer.
    """
    # Calculate intersection
    x1 = max(inner.x1, outer.x1)
    y1 = max(inner.y1, outer.y1)
    x2 = min(inner.x2, outer.x2)
    y2 = min(inner.y2, outer.y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    inner_area = inner.area
    
    return (intersection / inner_area) >= threshold if inner_area > 0 else False
