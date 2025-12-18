# USD Currency Detection with TensorFlow Lite

A quantitative study comparing different TensorFlow Lite models for real-time image detection of US Dollar bills on edge devices (Raspberry Pi 4). This project evaluates three state-of-the-art object detection models optimized for resource-constrained environments.

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![TensorFlow 2.8](https://img.shields.io/badge/TensorFlow-2.8-orange.svg)](https://www.tensorflow.org/)

## Overview

This research project investigates the performance of three TensorFlow Lite object detection models for identifying US currency denominations ($1, $5, $10, $20, $50, $100) on edge devices.

### Models Tested

1. **SSD-MobileNet-v2** - Balanced speed and accuracy
2. **SSD-MobileNet-v2-FPNLite-320x320** - Optimized for edge devices
3. **EfficientDet-d0** - Emphasis on accuracy with compound scaling

## Key Findings

| Model | Inference Time (ms) | FPS | mAP Score | Best For |
|-------|---------------------|-----|-----------|----------|
| **SSD-MobileNet-v2** | 170.21 | 2.7 | **43.59%** | Best overall accuracy |
| **SSD-MobileNet-v2-FPNLite** | **134.51** | 2.0 | 42.89% | **Best speed-accuracy balance** ⭐ |
| **EfficientDet-d0** | 1284.34 | 0.5 | 13.75% | Requires more powerful hardware |

### Recommended Model

**SSD-MobileNet-v2-FPNLite-320x320** is the optimal choice for edge deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/usd-currency-detection.git
cd usd-currency-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run real-time detection with webcam
python scripts/detection/TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite \
    --graph detect.tflite \
    --labels labelmap.txt \
    --threshold 0.5
```

## Dataset

- **Total Images**: ~3,000
- **Denominations**: $1, $5, $10, $20, $50, $100
- **Split**: 80% training, 10% validation, 10% testing

## Project Structure

```
usd-currency-detection/
├── data/                           # Dataset and annotations
├── models/                         # Trained TFLite models
├── scripts/                        # Python scripts
│   ├── data_preparation/           # Data processing scripts
│   ├── training/                   # Model training scripts
│   ├── detection/                  # Real-time detection
│   └── evaluation/                 # Evaluation scripts
├── configs/                        # Model configurations
├── docs/                           # Documentation
└── results/                        # Experimental results
```

## Results

### Accuracy by Denomination (SSD-MobileNet-v2-FPNLite)

| Denomination | Average mAP |
|--------------|-------------|
| $10 | 61.79% |
| $100 | 55.52% |
| $20 | 47.63% |
| $5 | 37.30% |
| $1 | 35.40% |
| $50 | 19.72% |

See `docs/research_paper.pdf` for complete analysis.

## Acknowledgments

- **Dr. Tanya Berger-Wolf** (Ohio State University) - Inspiration
- **Evan Juras** - TFLite deployment guides
- **Cartucho** - mAP calculation tool
- **Roboflow** - Dataset resources

## Contact

**Sadhil Mehta**
- Email: sadhil.mehta@gmail.com
- Linkedin: https://www.linkedin.com/in/sadhil-mehta/
- School: Carnegie Mellon University
