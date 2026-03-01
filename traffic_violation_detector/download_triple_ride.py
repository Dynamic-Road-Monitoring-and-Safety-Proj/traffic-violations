# -*- coding: utf-8 -*-
"""
Download script for triple ride detection dataset from Roboflow Universe.
Get your API key from: https://app.roboflow.com/settings/api
"""

import os
from roboflow import Roboflow

# Get API key from environment variable or prompt user
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    api_key = input("Enter your Roboflow API key: ").strip()
    if not api_key:
        print("[ERROR] API key is required!")
        exit(1)

print("="*60)
print("   Triple Ride Detection Dataset Downloader")
print("="*60)

try:
    rf = Roboflow(api_key=api_key)
    
    print("\n[*] Connecting to: harshas-workspace/triple-ride-detection")
    
    project = rf.workspace("harshas-workspace").project("triple-ride-detection")
    dataset = project.version(1).download("yolov8")
    
    print(f"\n[OK] Dataset downloaded successfully!")
    print(f"    Location: {dataset.location}")
    print(f"\nDataset ready for training!")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
