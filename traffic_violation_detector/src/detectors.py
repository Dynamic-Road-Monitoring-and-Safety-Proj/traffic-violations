"""
Individual detector classes for the traffic violation detection pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .models import BoundingBox, Detection
from .plate_ocr import LicensePlateRecognizer


logger = logging.getLogger("traffic_violation_detector")


class TwoWheelerDetector:
    """
    Detects motorcycles and scooters using YOLOv11.
    
    Uses the COCO pretrained model which has motorcycle class (index 3).
    """
    
    # COCO class indices for two-wheelers
    MOTORCYCLE_CLASS = 3
    
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.5,
        classes: Optional[list[int]] = None,
        device: str = "auto",
    ):
        """
        Initialize the two-wheeler detector.
        
        Args:
            model_path: Path to YOLOv11 model weights.
            confidence_threshold: Minimum confidence for detections.
            classes: List of class indices to detect. Defaults to [3] (motorcycle).
            device: Device to run inference on ('auto', 'cpu', 'cuda', '0', etc.).
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes or [self.MOTORCYCLE_CLASS]
        self.device = device
        
        logger.info(f"Loading two-wheeler detector from {model_path}")
        self.model = YOLO(str(model_path))
        
        # Get class names
        self.class_names = self.model.names
        logger.info(f"Two-wheeler detector loaded. Detecting classes: {[self.class_names[c] for c in self.classes]}")
    
    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """
        Detect two-wheelers in an image.
        
        Args:
            image: Input image in BGR format.
            
        Returns:
            List of BoundingBox objects for detected two-wheelers.
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            classes=self.classes,
            verbose=False,
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                bbox = BoundingBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    class_id=cls,
                    class_name=self.class_names[cls],
                )
                detections.append(bbox)
        
        logger.debug(f"Two-wheeler detector found {len(detections)} vehicles")
        return detections


class HelmetDetector:
    """
    Detects helmets and heads (no helmet) using a fine-tuned YOLOv8 model.
    
    The model has two classes:
    - head (class 0): Person without helmet (violation)
    - helmet (class 1): Person wearing helmet (compliant)
    """
    
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.45,
        device: str = "auto",
    ):
        """
        Initialize the helmet detector.
        
        Args:
            model_path: Path to helmet detection model weights.
            confidence_threshold: Minimum confidence for detections.
            device: Device to run inference on.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        logger.info(f"Loading helmet detector from {model_path}")
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        logger.info(f"Helmet detector loaded. Classes: {self.class_names}")
    
    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """
        Detect heads and helmets in an image.
        
        Args:
            image: Input image in BGR format.
            
        Returns:
            List of BoundingBox objects for detected heads/helmets.
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                bbox = BoundingBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    class_id=cls,
                    class_name=self.class_names[cls],
                )
                detections.append(bbox)
        
        logger.debug(f"Helmet detector found {len(detections)} heads/helmets")
        return detections
    
    def has_violation(self, image: np.ndarray) -> tuple[bool, list[BoundingBox]]:
        """
        Check if there's a helmet violation in the image.
        
        Args:
            image: Input image in BGR format (typically cropped vehicle region).
            
        Returns:
            Tuple of (has_violation, list of head detections without helmets).
        """
        detections = self.detect(image)
        
        # Filter for "head" detections (no helmet)
        # Class name might be "head", "Head", or similar depending on training
        head_detections = [
            d for d in detections 
            if d.class_name.lower() in ["head", "no_helmet", "nohelmet", "without helmet"]
        ]
        
        return len(head_detections) > 0, head_detections


class LicensePlateDetector:
    """
    Detects license plates using a fine-tuned YOLO model.
    """
    
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.4,
        device: str = "auto",
    ):
        """
        Initialize the license plate detector.
        
        Args:
            model_path: Path to license plate detection model weights.
            confidence_threshold: Minimum confidence for detections.
            device: Device to run inference on.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        logger.info(f"Loading license plate detector from {model_path}")
        self.model = YOLO(str(model_path))
        logger.info("License plate detector loaded")
    
    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image in BGR format.
            
        Returns:
            List of BoundingBox objects for detected license plates.
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                bbox = BoundingBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    class_id=cls,
                    class_name="license_plate",
                )
                detections.append(bbox)
        
        logger.debug(f"License plate detector found {len(detections)} plates")
        return detections


class PlateOCR:
    """
    OCR for license plate text recognition using fast-plate-ocr.
    """
    
    def __init__(
        self,
        model_name: str = "global-plates-mobile-vit-v2-model",
        device: str = "auto",
    ):
        """
        Initialize the plate OCR.
        
        Args:
            model_name: Name of the OCR model from fast-plate-ocr hub.
            device: Device to run inference on ('auto', 'cpu', 'cuda').
        """
        self.model_name = model_name
        
        # Map device to fast-plate-ocr format
        if device == "auto":
            ocr_device = "auto"
        elif device in ["cuda", "gpu"]:
            ocr_device = "cuda"
        else:
            ocr_device = "cpu"
        
        logger.info(f"Loading plate OCR model: {model_name}")
        self.recognizer = LicensePlateRecognizer(
            hub_ocr_model=model_name,
            device=ocr_device,
        )
        logger.info("Plate OCR loaded")
    
    def recognize(self, plate_image: np.ndarray) -> tuple[str, float]:
        """
        Recognize text from a license plate image.
        
        Args:
            plate_image: Cropped license plate image in BGR format.
            
        Returns:
            Tuple of (plate_text, average_confidence).
        """
        try:
            # Convert BGR to RGB - the CCT model expects RGB input
            if len(plate_image.shape) == 3 and plate_image.shape[2] == 3:
                rgb_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_plate = plate_image
            
            # Run OCR on the RGB image
            result = self.recognizer.run(rgb_plate, return_confidence=True)
            
            if isinstance(result, tuple):
                plates, confidences = result
                plate_text = plates[0] if plates else ""
                avg_confidence = float(np.mean(confidences[0])) if len(confidences) > 0 else 0.0
            else:
                plate_text = result[0] if result else ""
                avg_confidence = 0.0
            
            # Clean up the plate text (remove padding characters)
            plate_text = plate_text.replace("_", "").strip()
            
            logger.debug(f"OCR result: {plate_text} (conf: {avg_confidence:.3f})")
            return plate_text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0
