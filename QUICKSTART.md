# Quick Start Guide

Get up and running with USD currency detection in 5 minutes!

## 🚀 Option 1: Use Pre-trained Model (Fastest)

### Step 1: Install Dependencies
```bash
pip install tensorflow==2.8.0 opencv-python numpy pillow
```

### Step 2: Download Model
```bash
# Download the recommended model
# (Add your model to models/ssd_mobilenet_v2_fpnlite/)
```

### Step 3: Run Detection
```bash
python scripts/detection/TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite \
    --labels data/labelmap.txt
```

That's it! Point your webcam at US currency and see the detections.

## 🏗️ Option 2: Train Your Own Model

### Step 1: Prepare Dataset
```bash
# Organize your images
python scripts/data_preparation/train_val_test_split.py

# Convert to CSV
python scripts/data_preparation/create_csv.py

# Create TFRecords
python scripts/data_preparation/create_tfrecord.py \
    --csv_input=data/train_labels.csv \
    --image_dir=data/train \
    --labelmap=data/labelmap.txt \
    --output_path=data/train.tfrecord
```

### Step 2: Train (Google Colab recommended)
```bash
# See docs/TRAINING.md for detailed instructions
python model_main_tf2.py \
    --pipeline_config_path=configs/ssd_mobilenet_v2_fpnlite_pipeline.config \
    --model_dir=models/ssd_mobilenet_v2_fpnlite/checkpoint \
    --num_train_steps=40000
```

### Step 3: Convert to TFLite
```bash
python scripts/training/convert_to_tflite.py
```

### Step 4: Deploy
```bash
python scripts/detection/TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite
```

## 📊 Option 3: Evaluate Existing Model

### Calculate mAP Score
```bash
# Clone Cartucho's mAP tool
git clone https://github.com/Cartucho/mAP.git
cd mAP

# Run evaluation
python ../scripts/evaluation/calculate_map_cartucho.py \
    --labels ../data/labelmap.txt \
    --metric coco
```

### Benchmark Inference Speed
```bash
./scripts/evaluation/benchmark_inference.sh
```

## 🍓 Raspberry Pi Deployment

### Step 1: Install on Pi
```bash
# Install TFLite Runtime
pip3 install tflite-runtime==2.8.0 opencv-python numpy

# Transfer model
scp -r models/ pi@raspberrypi.local:~/
```

### Step 2: Run
```bash
python3 TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite \
    --resolution 640x480
```

## Common Commands

```bash
# List available models
ls models/

# Check model size
ls -lh models/*/detect.tflite

# Test camera
python -c "import cv2; print('OK' if cv2.VideoCapture(0).isOpened() else 'Error')"

# View labels
cat data/labelmap.txt
```

## Troubleshooting

**Camera not found:**
```bash
ls /dev/video*
```

**Low FPS:**
```bash
# Use lower resolution
--resolution 640x480

# Or try recommended model
--modeldir models/ssd_mobilenet_v2_fpnlite
```

**Import errors:**
```bash
# Reinstall dependencies
pip install --force-reinstall tensorflow==2.8.0
```

## Next Steps

- Read [TRAINING.md](docs/TRAINING.md) for detailed training guide
- See [EVALUATION.md](docs/EVALUATION.md) for evaluation metrics
- Check [DEPLOYMENT.md](docs/DEPLOYMENT.md) for Raspberry Pi setup
- View [research_paper.pdf](docs/research_paper.pdf) for full study

## Need Help?

- Check documentation in `docs/`
- Open an issue on GitHub
- Review existing issues for solutions

Happy detecting! 🎯
