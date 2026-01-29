"""
Plate OCR module - adapted from fast-plate-ocr.
https://github.com/ankandrew/fast-plate-ocr
"""

try:
    from .inference.plate_recognizer import LicensePlateRecognizer
    __all__ = ["LicensePlateRecognizer"]
except ImportError as e:
    # Handle case where onnxruntime is not installed
    import warnings
    warnings.warn(f"LicensePlateRecognizer not available: {e}")
    LicensePlateRecognizer = None
    __all__ = []
