# Evaluation Guide

This guide explains how to evaluate your trained TensorFlow Lite model's performance.

## Evaluation Metrics

### 1. mean Average Precision (mAP)

mAP measures detection accuracy at various Intersection over Union (IoU) thresholds.

#### Using Cartucho's mAP Tool

**Step 1: Clone the mAP repository**
```bash
git clone https://github.com/Cartucho/mAP.git
cd mAP
```

**Step 2: Prepare ground truth data**
```
mAP/
├── input/
│   ├── ground-truth/        # Your test annotations
│   │   ├── image1.txt
│   │   └── ...
│   └── detection-results/   # Model predictions
│       ├── image1.txt
│       └── ...
```

Ground truth format (one file per image):
```
1Dollar 100 150 200 250
10Dollar 300 350 400 450
```

Detection results format:
```
1Dollar 0.95 100 150 200 250
10Dollar 0.87 300 350 400 450
```

**Step 3: Run mAP calculation**
```bash
# Calculate mAP at multiple IoU thresholds
python ../scripts/evaluation/calculate_map_cartucho.py \
    --labels ../labelmap.txt \
    --metric coco \
    --outdir ../results/map_outputs
```

#### COCO Metric
Averages mAP across IoU thresholds 0.5:0.95:
```
mAP @ 0.50 IoU
mAP @ 0.55 IoU
...
mAP @ 0.95 IoU
Average: 42.89%
```

#### Pascal VOC Metric  
Uses single IoU threshold of 0.5:
```bash
python calculate_map_cartucho.py \
    --labels labelmap.txt \
    --metric pascalvoc
```

### 2. Inference Speed

Measures how fast the model processes images.

#### Using Google's Benchmark Tool

**Step 1: Download benchmark tool** (Raspberry Pi)
```bash
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_plus_flex

chmod +x linux_arm_benchmark_model_plus_flex
```

**Step 2: Run benchmark**
```bash
./linux_arm_benchmark_model_plus_flex \
    --graph=models/ssd_mobilenet_v2_fpnlite/detect.tflite \
    --num_threads=4 \
    --num_runs=50
```

**Output:**
```
Average inference time: 134.51 ms
Min: 128.3 ms
Max: 142.7 ms
```

#### Manual Timing (Python)
```python
import time
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Measure inference
times = []
for _ in range(100):
    start = time.time()
    interpreter.invoke()
    times.append((time.time() - start) * 1000)

print(f"Average: {sum(times)/len(times):.2f} ms")
```

### 3. Frames Per Second (FPS)

Measures real-time performance including preprocessing.

The detection script automatically displays FPS:
```bash
python scripts/detection/TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite
```

FPS is shown in top-left corner of video feed.

## Comparative Evaluation

### Running All Three Models

```bash
# Model 1: SSD-MobileNet-v2
python scripts/detection/TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2

# Model 2: SSD-MobileNet-v2-FPNLite  
python scripts/detection/TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite

# Model 3: EfficientDet-d0
python scripts/detection/TFLite_detection_webcam.py \
    --modeldir models/efficientdet_d0
```

Record metrics for each:
- Inference time (ms)
- FPS
- mAP score
- Per-class accuracy

## Results Analysis

### Expected Performance (Raspberry Pi 4)

| Model | Inference (ms) | FPS | mAP |
|-------|---------------|-----|-----|
| SSD-MobileNet-v2 | 170.21 | 2.7 | 43.59% |
| SSD-MobileNet-v2-FPNLite | 134.51 | 2.0 | 42.89% |
| EfficientDet-d0 | 1284.34 | 0.5 | 13.75% |

### Per-Denomination Accuracy

For SSD-MobileNet-v2-FPNLite:

| Denomination | mAP | Notes |
|--------------|-----|-------|
| $10 | 61.79% | Best performing |
| $100 | 55.52% | Strong performance |
| $20 | 47.63% | Good |
| $5 | 37.30% | Moderate |
| $1 | 35.40% | Needs improvement |
| $50 | 19.72% | Requires more training data |

### IoU Threshold Analysis

Create graphs showing mAP vs IoU threshold:

```python
import matplotlib.pyplot as plt

iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
map_scores = [82.62, 80.53, 80.53, 65.19, 61.63, 30.78, 17.50, 7.57, 2.58, 0.00]

plt.plot(iou_thresholds, map_scores, marker='o')
plt.xlabel('IoU Threshold')
plt.ylabel('mAP Score (%)')
plt.title('mAP vs IoU Threshold')
plt.grid(True)
plt.savefig('results/graphs/iou_threshold_map.png')
```

## Troubleshooting

### Low mAP Scores

**Possible causes:**
- Insufficient training data
- Poor quality annotations
- Model undertrained (< 40K steps)
- Test set too different from training set

**Solutions:**
- Add more diverse training images
- Review and correct annotations
- Train for more steps (50K+)
- Balance dataset across denominations

### Slow Inference

**Possible causes:**
- Model too complex for hardware
- CPU not optimized
- Running other processes

**Solutions:**
- Use FPNLite or MobileNet-v2 variants
- Close other applications
- Overclock Raspberry Pi (careful!)
- Use Coral Edge TPU accelerator

### Poor Real-time Performance

**Causes:**
- High-resolution webcam
- Display overhead
- OpenCV processing

**Solutions:**
```bash
# Lower resolution
--resolution 640x480  # Instead of 1280x720

# Disable display (headless mode)
--no_display

# Increase threshold (fewer detections)
--threshold 0.7  # Instead of 0.5
```

## Advanced Evaluation

### Confusion Matrix

Analyze which denominations are confused:
- $1 confused with $5 (similar colors)
- $20 confused with $10 (similar sizes)

### Precision-Recall Curves

Generate PR curves for each class using the mAP tool output.

### Error Analysis

Review false positives and false negatives:
```python
# Analyze detection errors
python scripts/evaluation/error_analysis.py \
    --model models/ssd_mobilenet_v2_fpnlite/detect.tflite \
    --test_dir data/test
```

## Citation

If using these evaluation methods in research, cite:

- **Cartucho mAP Tool**: https://github.com/Cartucho/mAP
- **TensorFlow Benchmark**: https://www.tensorflow.org/lite/performance/measurement
