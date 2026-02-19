"""
Data models for the traffic violation detection pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class ViolationType(Enum):
    """Types of traffic violations detected."""
    NO_HELMET = "no_helmet"
    THREE_SEATER = "three_seater"
    COMBINED = "no_helmet_and_three_seater"  # Both violations together
    

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 0.0
    class_id: int = -1
    class_name: str = ""
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def expand(self, factor: float) -> "BoundingBox":
        """Expand bounding box by a factor around its center."""
        cx, cy = self.center
        new_w = int(self.width * factor)
        new_h = int(self.height * factor)
        return BoundingBox(
            x1=max(0, cx - new_w // 2),
            y1=max(0, cy - new_h // 2),
            x2=cx + new_w // 2,
            y2=cy + new_h // 2,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )
    
    def crop_from_image(self, image: np.ndarray) -> np.ndarray:
        """Crop this bounding box region from an image."""
        h, w = image.shape[:2]
        x1 = max(0, self.x1)
        y1 = max(0, self.y1)
        x2 = min(w, self.x2)
        y2 = min(h, self.y2)
        return image[y1:y2, x1:x2]


@dataclass
class Detection:
    """Represents a single detection from any model."""
    bbox: BoundingBox
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class TwoWheelerDetection(Detection):
    """Detection of a two-wheeler (motorcycle/scooter)."""
    vehicle_type: str = "motorcycle"
    riders: list["RiderDetection"] = field(default_factory=list)


@dataclass
class RiderDetection:
    """Detection of a rider on a two-wheeler."""
    bbox: BoundingBox
    has_helmet: bool = True
    helmet_confidence: float = 0.0


@dataclass
class LicensePlateDetection(Detection):
    """Detection of a license plate."""
    plate_text: str = ""
    ocr_confidence: float = 0.0


@dataclass
class Violation:
    """Represents a traffic violation record."""
    violation_id: str
    violation_type: ViolationType
    timestamp: datetime
    license_plate: str
    plate_confidence: float
    vehicle_bbox: BoundingBox
    plate_bbox: Optional[BoundingBox]
    image_path: Optional[Path]
    source_frame: Optional[str]
    riders_count: int = 0  # Number of riders detected on the vehicle
    violation_details: str = ""  # Additional details about the violation
    
    def to_csv_row(self) -> dict:
        """Convert violation to CSV row format."""
        return {
            "violation_id": self.violation_id,
            "timestamp": self.timestamp.isoformat(),
            "violation_type": self.violation_type.value,
            "license_plate": self.license_plate,
            "plate_confidence": f"{self.plate_confidence:.3f}",
            "vehicle_bbox": f"{self.vehicle_bbox.to_tuple()}",
            "riders_count": str(self.riders_count),
            "violation_details": self.violation_details,
            "image_path": str(self.image_path) if self.image_path else "",
            "source_frame": self.source_frame or "",
        }


@dataclass
class ProcessingResult:
    """Result of processing a single frame/image."""
    frame_id: str
    timestamp: datetime
    two_wheelers_detected: int
    violations_found: int
    violations: list[Violation]
    processing_time_ms: float
    annotated_image: Optional[np.ndarray] = None
