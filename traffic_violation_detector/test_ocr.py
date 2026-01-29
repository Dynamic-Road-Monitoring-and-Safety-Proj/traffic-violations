#!/usr/bin/env python3
"""Test OCR functionality."""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.detectors import TwoWheelerDetector, HelmetDetector, LicensePlateDetector, PlateOCR

# Load image
image = cv2.imread('/Users/raghav_sarna/Desktop/Drive/Plaksha/Semester 6/ILGC/traffic_violation/img_test.jpg')
print(f'Image shape: {image.shape}')

# Initialize detectors
base_path = Path(__file__).parent
vehicle_det = TwoWheelerDetector(base_path / 'models/yolo11n.pt')
plate_det = LicensePlateDetector(base_path / 'models/license-plate-finetune-v1m.pt')
plate_ocr = PlateOCR()

# Detect vehicles
vehicles = vehicle_det.detect(image)
print(f'Vehicles found: {len(vehicles)}')

# For each vehicle, detect plates
for i, vehicle in enumerate(vehicles):
    vehicle_crop = image[vehicle.y1:vehicle.y2, vehicle.x1:vehicle.x2]
    plates = plate_det.detect(vehicle_crop)
    print(f'  Vehicle {i+1}: {len(plates)} plates found')
    
    for j, plate in enumerate(plates):
        plate_crop = vehicle_crop[plate.y1:plate.y2, plate.x1:plate.x2]
        print(f'    Plate {j+1} size: {plate_crop.shape}')
        text, conf = plate_ocr.recognize(plate_crop)
        print(f'    OCR result: "{text}" (confidence: {conf:.3f})')
