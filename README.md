# traffic-violations

A computer-vision pipeline for detecting traffic violations on two-wheelers — currently focused on helmet-less riders, with license-plate OCR for identification. Part of the [SadakVision](https://sadakvision.com) road-safety platform, developed by the [Dynamic-Road-Monitoring-and-Safety-Proj](https://github.com/Dynamic-Road-Monitoring-and-Safety-Proj) team with Prof. Srikant Srinivasan's Dixon IoT Lab under an MoU with Punjab Police.

## How it fits into SadakVision

SadakVision combines an SDG-11-aligned mobile/IoT sensing layer with AI-driven video analysis to study driver behavior, congestion, road quality, and weather — at roughly 73% lower cost than comparable state-of-the-art monitoring alternatives. This repo is the enforcement-side complement to that platform: it takes traffic camera / dashcam footage and flags helmet violations with an identifiable license plate, so violations can be logged and reported alongside the road-quality and congestion data collected by the [`Sensors_prod`](https://github.com/Dynamic-Road-Monitoring-and-Safety-Proj/Sensors_prod) and [`RS-TM`](https://github.com/Dynamic-Road-Monitoring-and-Safety-Proj/RS-TM) Android apps, and visualized in the [`web_v2`](https://github.com/Dynamic-Road-Monitoring-and-Safety-Proj/web_v2) dashboard.

## What's inside

The actual pipeline lives in [`traffic_violation_detector/`](./traffic_violation_detector) — see its [README](./traffic_violation_detector/README.md) for setup, usage, and architecture details. In short, it:

- Detects two-wheelers with a YOLOv11 model
- Runs a custom-trained YOLOv8 model to detect riders without helmets
- Localizes and reads license plates with a fine-tuned YOLO model + `fast-plate-ocr`
- Logs violations (timestamp, plate number, annotated frame) to CSV, with output ready for review/reporting

## Tech Stack

- **Language**: Python
- **Computer Vision**: YOLOv11 / YOLOv8 (Ultralytics), fine-tuned license-plate detector
- **OCR**: `fast-plate-ocr`
- **Packaging**: Docker / docker-compose (AWS ECS, Lambda, or EC2 ready)

## Links

- Live platform: [sadakvision.com](https://sadakvision.com)
- Org: [Dynamic-Road-Monitoring-and-Safety-Proj](https://github.com/Dynamic-Road-Monitoring-and-Safety-Proj)
