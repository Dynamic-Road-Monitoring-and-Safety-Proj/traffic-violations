#!/usr/bin/env python3
"""Simple test runner for the traffic violation detector."""

import sys
from pathlib import Path

# Add the parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    main()
