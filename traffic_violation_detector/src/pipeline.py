"""
Main pipeline for traffic violation detection.

This module orchestrates the entire detection pipeline:
1. Two-wheeler (motorcycle/scooter) detection
2. Helmet violation detection
3. License plate detection
4. License plate OCR
5. Violation logging
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import PipelineConfig, load_config
from .detectors import (
    TwoWheelerDetector,
    HelmetDetector,
    LicensePlateDetector,
    PlateOCR,
)
from .models import (
    BoundingBox,
    ProcessingResult,
    Violation,
    ViolationType,
)
from .utils import (
    ViolationCSVWriter,
    annotate_violation,
    ensure_directories,
    generate_violation_id,
    is_bbox_inside,
    save_violation_image,
    setup_logging,
)


logger = logging.getLogger("traffic_violation_detector")


class TrafficViolationPipeline:
    """
    Complete pipeline for detecting traffic violations.
    
    The pipeline performs the following steps:
    1. Detect two-wheelers (motorcycles/scooters) in the frame
    2. For each two-wheeler, check for helmet violations
    3. If violation found, detect and read license plate
    4. Log violation with plate number to CSV
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize the traffic violation detection pipeline.
        
        Args:
            config: Pre-loaded PipelineConfig object.
            config_path: Path to configuration YAML file.
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.logging.level,
            log_file=self.config.get_log_file_path(),
            log_format=self.config.logging.format,
        )
        
        # Ensure output directories exist
        ensure_directories([
            self.config.get_output_dir(),
            self.config.get_images_dir(),
            self.config.get_log_file_path().parent,
        ])
        
        # Initialize CSV writer
        self.csv_writer = ViolationCSVWriter(self.config.get_violations_csv_path())
        
        # Initialize detectors (lazy loading for AWS Lambda compatibility)
        self._two_wheeler_detector: Optional[TwoWheelerDetector] = None
        self._helmet_detector: Optional[HelmetDetector] = None
        self._plate_detector: Optional[LicensePlateDetector] = None
        self._plate_ocr: Optional[PlateOCR] = None
        
        self.logger.info("Traffic Violation Pipeline initialized")
        self.logger.info(f"Output directory: {self.config.get_output_dir()}")
        self.logger.info(f"Violations CSV: {self.config.get_violations_csv_path()}")
    
    @property
    def two_wheeler_detector(self) -> TwoWheelerDetector:
        """Lazy-load two-wheeler detector."""
        if self._two_wheeler_detector is None:
            self._two_wheeler_detector = TwoWheelerDetector(
                model_path=self.config.get_model_path("vehicle_detector"),
                confidence_threshold=self.config.detection.thresholds["vehicle"],
                classes=self.config.detection.two_wheeler_classes,
            )
        return self._two_wheeler_detector
    
    @property
    def helmet_detector(self) -> HelmetDetector:
        """Lazy-load helmet detector."""
        if self._helmet_detector is None:
            self._helmet_detector = HelmetDetector(
                model_path=self.config.get_model_path("helmet_detector"),
                confidence_threshold=self.config.detection.thresholds["helmet"],
            )
        return self._helmet_detector
    
    @property
    def plate_detector(self) -> LicensePlateDetector:
        """Lazy-load plate detector."""
        if self._plate_detector is None:
            self._plate_detector = LicensePlateDetector(
                model_path=self.config.get_model_path("plate_detector"),
                confidence_threshold=self.config.detection.thresholds["plate"],
            )
        return self._plate_detector
    
    @property
    def plate_ocr(self) -> PlateOCR:
        """Lazy-load plate OCR."""
        if self._plate_ocr is None:
            self._plate_ocr = PlateOCR(
                model_name=self.config.models.plate_ocr,
            )
        return self._plate_ocr
    
    def _expand_bbox_to_frame(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int],
        expansion_factor: float,
    ) -> BoundingBox:
        """
        Expand bounding box while keeping it within frame bounds.
        
        Args:
            bbox: Original bounding box.
            frame_shape: (height, width) of the frame.
            expansion_factor: Factor to expand the box by.
            
        Returns:
            Expanded bounding box within frame bounds.
        """
        h, w = frame_shape
        expanded = bbox.expand(expansion_factor)
        
        return BoundingBox(
            x1=max(0, expanded.x1),
            y1=max(0, expanded.y1),
            x2=min(w, expanded.x2),
            y2=min(h, expanded.y2),
            confidence=bbox.confidence,
            class_id=bbox.class_id,
            class_name=bbox.class_name,
        )
    
    def _find_head_for_vehicle(
        self,
        vehicle_bbox: BoundingBox,
        head_detections: list[BoundingBox],
        frame_height: int,
    ) -> list[BoundingBox]:
        """
        Find heads (no helmet) that belong to a vehicle based on spatial proximity.
        
        A head is associated with a vehicle if:
        - It's above or overlapping with the top portion of the vehicle
        - It's horizontally aligned with the vehicle
        
        Args:
            vehicle_bbox: Bounding box of the vehicle.
            head_detections: List of all detected heads in the frame.
            frame_height: Height of the frame for normalization.
            
        Returns:
            List of head bounding boxes associated with this vehicle.
        """
        associated_heads = []
        
        vehicle_center_x = (vehicle_bbox.x1 + vehicle_bbox.x2) // 2
        vehicle_width = vehicle_bbox.width
        vehicle_top = vehicle_bbox.y1
        
        for head in head_detections:
            head_center_x = (head.x1 + head.x2) // 2
            head_bottom = head.y2
            
            # Check horizontal alignment: head center should be within vehicle width
            horizontal_margin = vehicle_width * 0.6  # Allow some margin
            is_horizontally_aligned = abs(head_center_x - vehicle_center_x) < horizontal_margin
            
            # Check vertical position: head should be above or at the top of vehicle
            # Head bottom should be near or above the vehicle top (with margin for overlap)
            vertical_margin = vehicle_bbox.height * 0.5
            is_above_vehicle = head_bottom <= vehicle_top + vertical_margin
            
            if is_horizontally_aligned and is_above_vehicle:
                associated_heads.append(head)
        
        return associated_heads
    
    def _find_plate_for_vehicle(
        self,
        vehicle_bbox: BoundingBox,
        plate_detections: list[BoundingBox],
    ) -> Optional[BoundingBox]:
        """
        Find the license plate that belongs to a vehicle.
        
        Args:
            vehicle_bbox: Bounding box of the vehicle.
            plate_detections: List of all detected plates in the frame.
            
        Returns:
            The plate bounding box if found, None otherwise.
        """
        best_plate = None
        best_overlap = 0.0
        
        for plate in plate_detections:
            # Check if plate is inside or near the vehicle
            if is_bbox_inside(plate, vehicle_bbox, threshold=0.3):
                overlap = (plate.confidence + 1.0)  # Prioritize by confidence
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_plate = plate
        
        # If no plate inside vehicle, check for plates below the vehicle
        if best_plate is None:
            vehicle_bottom = vehicle_bbox.y2
            vehicle_center_x = (vehicle_bbox.x1 + vehicle_bbox.x2) // 2
            
            for plate in plate_detections:
                plate_center_x = (plate.x1 + plate.x2) // 2
                
                # Plate should be near the bottom of vehicle and horizontally aligned
                if (plate.y1 >= vehicle_bbox.y1 and 
                    plate.y2 <= vehicle_bottom + vehicle_bbox.height // 2 and
                    abs(plate_center_x - vehicle_center_x) < vehicle_bbox.width // 2):
                    
                    if plate.confidence > best_overlap:
                        best_overlap = plate.confidence
                        best_plate = plate
        
        return best_plate
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str = "",
        source_name: str = "",
    ) -> ProcessingResult:
        """
        Process a single frame for traffic violations.
        
        Args:
            frame: Input frame in BGR format.
            frame_id: Unique identifier for the frame.
            source_name: Name of the source (file, camera, etc.).
            
        Returns:
            ProcessingResult with all detected violations.
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        if not frame_id:
            frame_id = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        violations = []
        frame_height, frame_width = frame.shape[:2]
        
        # Step 1: Detect two-wheelers
        self.logger.debug("Step 1: Detecting two-wheelers...")
        vehicle_detections = self.two_wheeler_detector.detect(frame)
        self.logger.info(f"Found {len(vehicle_detections)} two-wheelers")
        
        if not vehicle_detections:
            processing_time = (time.time() - start_time) * 1000
            return ProcessingResult(
                frame_id=frame_id,
                timestamp=timestamp,
                two_wheelers_detected=0,
                violations_found=0,
                violations=[],
                processing_time_ms=processing_time,
            )
        
        # Step 2: Detect license plates in the entire frame
        # (more efficient than per-vehicle detection)
        self.logger.debug("Step 2: Detecting license plates...")
        all_plate_detections = self.plate_detector.detect(frame)
        self.logger.info(f"Found {len(all_plate_detections)} license plates")
        
        # Step 3: Detect all heads/helmets in the full frame
        # This is more reliable than per-vehicle cropping since heads may be outside vehicle bbox
        self.logger.debug("Step 3: Detecting helmets in full frame...")
        all_helmet_detections = self.helmet_detector.detect(frame)
        
        # Filter for "head" class only (no helmet = violation)
        head_detections = [
            d for d in all_helmet_detections
            if d.class_name.lower() in ["head", "no_helmet", "nohelmet", "without helmet"]
        ]
        self.logger.info(f"Found {len(head_detections)} heads without helmets in frame")
        
        # Step 4: Process each two-wheeler
        for vehicle_idx, vehicle_bbox in enumerate(vehicle_detections):
            self.logger.debug(f"Processing vehicle {vehicle_idx + 1}/{len(vehicle_detections)}")
            
            # Step 4a: Find heads (no helmet) associated with this vehicle
            associated_heads = self._find_head_for_vehicle(
                vehicle_bbox, head_detections, frame_height
            )
            
            has_violation = len(associated_heads) > 0
            
            if not has_violation:
                self.logger.debug(f"Vehicle {vehicle_idx + 1}: No violation detected")
                continue
            
            self.logger.info(f"Vehicle {vehicle_idx + 1}: HELMET VIOLATION DETECTED! ({len(associated_heads)} riders without helmet)")
            
            # Step 4b: Find license plate for this vehicle
            plate_bbox = self._find_plate_for_vehicle(vehicle_bbox, all_plate_detections)
            
            plate_text = ""
            plate_confidence = 0.0
            
            if plate_bbox is not None:
                # Step 4c: Perform OCR on the license plate
                self.logger.debug("Running OCR on license plate...")
                plate_crop = plate_bbox.crop_from_image(frame)
                
                if plate_crop.size > 0:
                    plate_text, plate_confidence = self.plate_ocr.recognize(plate_crop)
                    self.logger.info(f"License plate: {plate_text} (conf: {plate_confidence:.3f})")
            else:
                self.logger.warning("No license plate found for violating vehicle")
            
            # Create violation record
            violation_id = generate_violation_id()
            
            # Annotate and save image if configured
            image_path = None
            if self.config.output.save_images:
                annotated = annotate_violation(
                    frame,
                    vehicle_bbox,
                    plate_bbox,
                    plate_text,
                    "NO HELMET",
                )
                image_path = save_violation_image(
                    annotated,
                    self.config.get_images_dir(),
                    violation_id,
                )
            
            violation = Violation(
                violation_id=violation_id,
                violation_type=ViolationType.NO_HELMET,
                timestamp=timestamp,
                license_plate=plate_text,
                plate_confidence=plate_confidence,
                vehicle_bbox=vehicle_bbox,
                plate_bbox=plate_bbox,
                image_path=image_path,
                source_frame=source_name,
            )
            
            violations.append(violation)
            
            # Write to CSV immediately
            self.csv_writer.write_violation(violation)
            self.logger.info(f"Violation recorded: {violation_id} - Plate: {plate_text}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            two_wheelers_detected=len(vehicle_detections),
            violations_found=len(violations),
            violations=violations,
            processing_time_ms=processing_time,
        )
    
    def process_image(self, image_path: str | Path) -> ProcessingResult:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            ProcessingResult with detected violations.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.logger.info(f"Processing image: {image_path}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        return self.process_frame(
            frame,
            frame_id=image_path.stem,
            source_name=str(image_path),
        )
    
    def process_video(
        self,
        video_path: str | Path,
        frame_skip: int = 5,
        max_frames: Optional[int] = None,
        show_preview: bool = False,
    ) -> list[ProcessingResult]:
        """
        Process a video file.
        
        Args:
            video_path: Path to the video file.
            frame_skip: Process every Nth frame.
            max_frames: Maximum number of frames to process.
            show_preview: Show preview window during processing.
            
        Returns:
            List of ProcessingResult for each processed frame.
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Video info: {total_frames} frames @ {fps:.2f} FPS")
        
        results = []
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    result = self.process_frame(
                        frame,
                        frame_id=f"{video_path.stem}_frame_{frame_count:06d}",
                        source_name=str(video_path),
                    )
                    results.append(result)
                    processed_count += 1
                    
                    if result.violations_found > 0:
                        self.logger.info(
                            f"Frame {frame_count}: {result.violations_found} violations found"
                        )
                    
                    if show_preview:
                        cv2.imshow("Traffic Violation Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if max_frames and processed_count >= max_frames:
                        break
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 100 == 0:
                    self.logger.info(f"Progress: {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Summary
        total_violations = sum(r.violations_found for r in results)
        self.logger.info(f"Video processing complete: {processed_count} frames processed")
        self.logger.info(f"Total violations found: {total_violations}")
        
        return results
    
    def process_directory(
        self,
        directory_path: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list[ProcessingResult]:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to the directory.
            extensions: Tuple of valid image extensions.
            
        Returns:
            List of ProcessingResult for each processed image.
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory_path.glob(f"*{ext}"))
            image_paths.extend(directory_path.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(set(image_paths))
        
        self.logger.info(f"Found {len(image_paths)} images in {directory_path}")
        
        results = []
        for idx, image_path in enumerate(image_paths):
            self.logger.info(f"Processing {idx + 1}/{len(image_paths)}: {image_path.name}")
            
            try:
                result = self.process_image(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
        
        # Summary
        total_violations = sum(r.violations_found for r in results)
        self.logger.info(f"Directory processing complete: {len(results)} images processed")
        self.logger.info(f"Total violations found: {total_violations}")
        
        return results
