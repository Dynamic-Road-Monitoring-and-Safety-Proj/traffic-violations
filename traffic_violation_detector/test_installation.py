"""
Test script to verify the traffic violation detector installation.
Run this after installing dependencies to ensure everything works.
"""

import sys
from pathlib import Path


def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    errors = []
    
    # Test core dependencies
    try:
        import cv2
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ OpenCV: {e}")
    
    try:
        import numpy as np
        print(f"  ✓ NumPy: {np.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ NumPy: {e}")
    
    try:
        import yaml
        print(f"  ✓ PyYAML: OK")
    except ImportError as e:
        errors.append(f"  ✗ PyYAML: {e}")
    
    try:
        from ultralytics import YOLO
        print(f"  ✓ Ultralytics YOLO: OK")
    except ImportError as e:
        errors.append(f"  ✗ Ultralytics: {e}")
    
    try:
        import onnxruntime as ort
        print(f"  ✓ ONNX Runtime: {ort.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ ONNX Runtime: {e}")
    
    return errors


def test_project_structure():
    """Test project structure and files."""
    print("\nTesting project structure...")
    
    project_root = Path(__file__).parent
    
    required_files = [
        "config/config.yaml",
        "models/yolo11n.pt",
        "models/helmet_detector.pt",
        "models/license-plate-finetune-v1m.pt",
        "src/pipeline.py",
        "src/detectors.py",
        "requirements.txt",
    ]
    
    errors = []
    
    for file in required_files:
        filepath = project_root / file
        if filepath.exists():
            print(f"  ✓ {file}")
        else:
            errors.append(f"  ✗ {file} - NOT FOUND")
    
    return errors


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        # Add src to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from src.config import load_config
        config_path = project_root / "config" / "config.yaml"
        config = load_config(config_path)
        
        print(f"  ✓ Configuration loaded successfully")
        print(f"    - Vehicle detector: {config.models.vehicle_detector}")
        print(f"    - Helmet detector: {config.models.helmet_detector}")
        print(f"    - Plate detector: {config.models.plate_detector}")
        print(f"    - OCR model: {config.models.plate_ocr}")
        
        return []
    except Exception as e:
        return [f"  ✗ Configuration loading failed: {e}"]


def test_model_loading():
    """Test loading the detection models."""
    print("\nTesting model loading (this may take a moment)...")
    
    errors = []
    
    try:
        from ultralytics import YOLO
        
        project_root = Path(__file__).parent
        
        # Test YOLOv11 (vehicle detector)
        vehicle_model_path = project_root / "models" / "yolo11n.pt"
        if vehicle_model_path.exists():
            model = YOLO(str(vehicle_model_path))
            print(f"  ✓ Vehicle detector loaded")
        else:
            errors.append(f"  ✗ Vehicle detector model not found")
        
        # Test Helmet detector
        helmet_model_path = project_root / "models" / "helmet_detector.pt"
        if helmet_model_path.exists():
            model = YOLO(str(helmet_model_path))
            print(f"  ✓ Helmet detector loaded")
        else:
            errors.append(f"  ✗ Helmet detector model not found")
        
        # Test Plate detector
        plate_model_path = project_root / "models" / "license-plate-finetune-v1m.pt"
        if plate_model_path.exists():
            model = YOLO(str(plate_model_path))
            print(f"  ✓ Plate detector loaded")
        else:
            errors.append(f"  ✗ Plate detector model not found")
            
    except Exception as e:
        errors.append(f"  ✗ Model loading error: {e}")
    
    return errors


def main():
    """Run all tests."""
    print("=" * 60)
    print("Traffic Violation Detector - Installation Test")
    print("=" * 60)
    
    all_errors = []
    
    # Run tests
    all_errors.extend(test_imports())
    all_errors.extend(test_project_structure())
    all_errors.extend(test_config_loading())
    all_errors.extend(test_model_loading())
    
    # Summary
    print("\n" + "=" * 60)
    if all_errors:
        print("SOME TESTS FAILED:")
        for error in all_errors:
            print(error)
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("ALL TESTS PASSED! ✓")
        print("\nYou can now run the pipeline:")
        print("  python -m traffic_violation_detector --help")
        return 0


if __name__ == "__main__":
    sys.exit(main())
