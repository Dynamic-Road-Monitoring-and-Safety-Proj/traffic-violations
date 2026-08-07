# Traffic Violation Detection System

> Part of [SadakVision](https://sadakvision.com) — see the [repo root README](../README.md) for how this fits into the broader platform.

A complete computer vision pipeline for detecting traffic violations on two-wheelers (motorcycles and scooters), specifically focusing on helmet violations. The system detects violating vehicles, reads their license plates using OCR, and logs violations to a CSV file.

## 🎯 Features

- **Two-Wheeler Detection**: Uses YOLOv11 to detect motorcycles and scooters
- **Helmet Violation Detection**: Custom-trained YOLOv8 model to detect riders without helmets
- **License Plate Detection**: Fine-tuned YOLO model for accurate plate localization
- **License Plate OCR**: Fast and accurate OCR using the fast-plate-ocr library
- **Violation Logging**: Automatic logging of violations with timestamps, plate numbers, and annotated images
- **AWS Ready**: Designed for easy deployment on AWS (ECS, Lambda, EC2)

## 📁 Project Structure

```
traffic_violation_detector/
├── __init__.py           # Package initialization
├── __main__.py           # CLI entry point
├── config/
│   └── config.yaml       # Pipeline configuration
├── models/
│   ├── yolo11n.pt       # YOLOv11 for vehicle detection
│   ├── helmet_detector.pt # Helmet detection model
│   └── license-plate-finetune-v1m.pt  # Plate detection model
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration loader
│   ├── detectors.py      # Individual detector classes
│   ├── main.py          # CLI implementation
│   ├── models.py        # Data models
│   ├── pipeline.py      # Main processing pipeline
│   ├── utils.py         # Utility functions
│   └── plate_ocr/       # License plate OCR module
├── output/              # Detection results
├── logs/                # Application logs
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container build file
└── docker-compose.yml  # Container orchestration
```

## 🚀 Quick Start

### Installation

1. **Clone or copy the project**:
   ```bash
   cd traffic_violation_detector
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Process a Single Image
```bash
python -m traffic_violation_detector --image path/to/image.jpg
```

#### Process a Directory of Images
```bash
python -m traffic_violation_detector --directory path/to/images/
```

#### Process a Video File
```bash
python -m traffic_violation_detector --video path/to/video.mp4 --frame-skip 5
```

#### With Preview Window (for video)
```bash
python -m traffic_violation_detector --video path/to/video.mp4 --preview
```

### Python API

```python
from traffic_violation_detector import TrafficViolationPipeline

# Initialize the pipeline
pipeline = TrafficViolationPipeline()

# Process a single image
result = pipeline.process_image("path/to/image.jpg")
print(f"Violations found: {result.violations_found}")
for violation in result.violations:
    print(f"  - Plate: {violation.license_plate}")

# Process a video
results = pipeline.process_video("path/to/video.mp4", frame_skip=5)

# Process a directory
results = pipeline.process_directory("path/to/images/")
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize the pipeline:

```yaml
# Detection thresholds
detection:
  thresholds:
    vehicle: 0.5    # Confidence for vehicle detection
    helmet: 0.45    # Confidence for helmet detection
    plate: 0.4      # Confidence for plate detection

# Output settings
output:
  output_dir: "output"
  violations_csv: "violations.csv"
  save_images: true
```

## 🐳 Docker Deployment

### Build the Image
```bash
docker build -t traffic-violation-detector:latest .
```

### Run with Docker
```bash
docker run -v $(pwd)/input:/app/input:ro \
           -v $(pwd)/output:/app/output \
           traffic-violation-detector:latest \
           --directory /app/input
```

### Using Docker Compose
```bash
# CPU version
docker-compose up traffic-detector

# GPU version (requires nvidia-docker)
docker-compose up traffic-detector-gpu
```

## ☁️ AWS Deployment

### EC2 Deployment

1. Launch an EC2 instance (recommended: `g4dn.xlarge` for GPU)
2. Install Docker and nvidia-docker (for GPU)
3. Pull and run the container

### ECS Deployment

1. Push the Docker image to ECR
2. Create an ECS task definition
3. Configure the service with appropriate resources

### Lambda Deployment (for image processing)

For Lambda deployment, consider:
- Using Lambda container images
- Mounting EFS for model storage
- Setting memory to 3008MB+ for ML workloads

## 📊 Output Format

### CSV Structure

The violations are logged to `output/violations.csv`:

| Column | Description |
|--------|-------------|
| violation_id | Unique identifier |
| timestamp | ISO format timestamp |
| violation_type | Type of violation (e.g., "no_helmet") |
| license_plate | Recognized plate text |
| plate_confidence | OCR confidence score |
| vehicle_bbox | Vehicle bounding box coordinates |
| image_path | Path to annotated image |
| source_frame | Source file/frame name |

### Annotated Images

Violation images are saved to `output/violation_images/` with:
- Red bounding box around the violating vehicle
- Yellow bounding box around the license plate
- Text overlay with violation type and plate number

## 🔧 Model Information

| Model | Purpose | Source |
|-------|---------|--------|
| YOLOv11n | Two-wheeler detection | Ultralytics (COCO pretrained) |
| Helmet Detector | Helmet violation detection | Custom trained YOLOv8 |
| License Plate Detector | Plate localization | Fine-tuned YOLO |
| Plate OCR | Text recognition | fast-plate-ocr (global model) |

## 📝 License

This project uses models and libraries with their respective licenses:
- Ultralytics YOLO: AGPL-3.0
- fast-plate-ocr: MIT License
