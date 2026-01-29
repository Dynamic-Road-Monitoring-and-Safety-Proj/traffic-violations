"""
Debug script to visualize all detection stages.
Saves annotated images showing bounding boxes for each detection step.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.detectors import TwoWheelerDetector, HelmetDetector, LicensePlateDetector
from src.models import BoundingBox
from src.utils import load_image


def draw_bbox_with_label(image, bbox, color, label, thickness=2):
    """Draw bounding box with label on image."""
    cv2.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
    
    cv2.rectangle(
        image,
        (bbox.x1, bbox.y1 - text_h - baseline - 5),
        (bbox.x1 + text_w + 5, bbox.y1),
        color,
        -1
    )
    cv2.putText(
        image,
        label,
        (bbox.x1 + 2, bbox.y1 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        1
    )
    return image


def debug_detections(image_path: str, output_dir: str = "debug_output"):
    """Run detection pipeline and save debug images."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    config = load_config(config_path)
    
    # Load image
    print(f"Loading image: {image_path}")
    original_image = load_image(image_path)
    print(f"Image size: {original_image.shape}")
    
    # ==================== Step 1: Two-wheeler detection ====================
    print("\n=== Step 1: Two-Wheeler Detection ===")
    two_wheeler_detector = TwoWheelerDetector(
        model_path=config.get_model_path("vehicle_detector"),
        confidence_threshold=config.detection.thresholds["vehicle"],
        classes=config.detection.two_wheeler_classes,
    )
    
    vehicle_detections = two_wheeler_detector.detect(original_image)
    print(f"Found {len(vehicle_detections)} two-wheelers")
    
    # Draw vehicle detections
    vehicle_img = original_image.copy()
    for i, bbox in enumerate(vehicle_detections):
        label = f"Vehicle {i+1}: {bbox.class_name} ({bbox.confidence:.2f})"
        print(f"  {label} at ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
        vehicle_img = draw_bbox_with_label(vehicle_img, bbox, (0, 255, 0), label, 3)
    
    cv2.imwrite(str(output_dir / "1_vehicles.jpg"), vehicle_img)
    print(f"Saved: {output_dir / '1_vehicles.jpg'}")
    
    # ==================== Step 2: License plate detection ====================
    print("\n=== Step 2: License Plate Detection ===")
    plate_detector = LicensePlateDetector(
        model_path=config.get_model_path("plate_detector"),
        confidence_threshold=config.detection.thresholds["plate"],
    )
    
    plate_detections = plate_detector.detect(original_image)
    print(f"Found {len(plate_detections)} license plates")
    
    # Draw plate detections
    plate_img = original_image.copy()
    for i, bbox in enumerate(plate_detections):
        label = f"Plate {i+1} ({bbox.confidence:.2f})"
        print(f"  {label} at ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
        plate_img = draw_bbox_with_label(plate_img, bbox, (0, 255, 255), label, 2)
    
    cv2.imwrite(str(output_dir / "2_plates.jpg"), plate_img)
    print(f"Saved: {output_dir / '2_plates.jpg'}")
    
    # ==================== Step 3: Helmet detection (on full image) ====================
    print("\n=== Step 3: Helmet Detection (Full Image) ===")
    helmet_detector = HelmetDetector(
        model_path=config.get_model_path("helmet_detector"),
        confidence_threshold=config.detection.thresholds["helmet"],
    )
    
    # Run on full image first
    full_helmet_detections = helmet_detector.detect(original_image)
    print(f"Found {len(full_helmet_detections)} heads/helmets in full image")
    
    helmet_img = original_image.copy()
    for i, bbox in enumerate(full_helmet_detections):
        # Color based on class: red for head (violation), green for helmet
        if bbox.class_name.lower() == "head":
            color = (0, 0, 255)  # Red for no helmet
        elif bbox.class_name.lower() == "helmet":
            color = (0, 255, 0)  # Green for helmet
        else:
            color = (255, 0, 0)  # Blue for other
        
        label = f"{bbox.class_name} ({bbox.confidence:.2f})"
        print(f"  {label} at ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
        helmet_img = draw_bbox_with_label(helmet_img, bbox, color, label, 2)
    
    cv2.imwrite(str(output_dir / "3_helmets_full.jpg"), helmet_img)
    print(f"Saved: {output_dir / '3_helmets_full.jpg'}")
    
    # ==================== Step 4: Per-vehicle helmet detection ====================
    print("\n=== Step 4: Per-Vehicle Helmet Detection ===")
    
    combined_img = original_image.copy()
    
    for i, vehicle_bbox in enumerate(vehicle_detections):
        print(f"\n--- Vehicle {i+1} ---")
        
        # Expand vehicle bbox
        h, w = original_image.shape[:2]
        expansion_factor = config.processing.roi_expansion
        expanded = vehicle_bbox.expand(expansion_factor)
        expanded = BoundingBox(
            x1=max(0, expanded.x1),
            y1=max(0, expanded.y1),
            x2=min(w, expanded.x2),
            y2=min(h, expanded.y2),
            confidence=vehicle_bbox.confidence,
            class_id=vehicle_bbox.class_id,
            class_name=vehicle_bbox.class_name,
        )
        
        # Crop vehicle region
        vehicle_crop = expanded.crop_from_image(original_image)
        print(f"  Vehicle crop size: {vehicle_crop.shape}")
        
        # Detect helmets in crop
        helmet_detections = helmet_detector.detect(vehicle_crop)
        print(f"  Found {len(helmet_detections)} detections in vehicle crop")
        
        # Check for violations
        has_violation = False
        for det in helmet_detections:
            print(f"    - {det.class_name} ({det.confidence:.2f})")
            if det.class_name.lower() in ["head", "no_helmet", "nohelmet", "without helmet"]:
                has_violation = True
        
        # Draw on combined image
        if has_violation:
            color = (0, 0, 255)  # Red for violation
            label = f"Vehicle {i+1}: VIOLATION"
        else:
            color = (0, 255, 0)  # Green for OK
            label = f"Vehicle {i+1}: OK"
        
        combined_img = draw_bbox_with_label(combined_img, vehicle_bbox, color, label, 3)
        
        # Save individual vehicle crop with detections
        vehicle_debug = vehicle_crop.copy()
        for det in helmet_detections:
            det_color = (0, 0, 255) if det.class_name.lower() == "head" else (0, 255, 0)
            vehicle_debug = draw_bbox_with_label(
                vehicle_debug, det, det_color, 
                f"{det.class_name} ({det.confidence:.2f})", 2
            )
        cv2.imwrite(str(output_dir / f"4_vehicle_{i+1}_crop.jpg"), vehicle_debug)
    
    cv2.imwrite(str(output_dir / "5_combined_result.jpg"), combined_img)
    print(f"\nSaved: {output_dir / '5_combined_result.jpg'}")
    
    # ==================== Summary ====================
    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    print(f"Two-wheelers detected: {len(vehicle_detections)}")
    print(f"License plates detected: {len(plate_detections)}")
    print(f"Heads/Helmets in full image: {len(full_helmet_detections)}")
    print(f"\nDebug images saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Debug detection pipeline")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--output", "-o", default="debug_output", help="Output directory")
    args = parser.parse_args()
    
    debug_detections(args.image, args.output)
