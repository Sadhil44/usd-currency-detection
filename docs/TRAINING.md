# Training Guide

This guide explains how to train your own TensorFlow Lite object detection model for USD currency detection.

## Prerequisites

- Google Colab (recommended) or powerful local machine with GPU
- TensorFlow 2.8.0
- TensorFlow Object Detection API
- Labeled dataset with Pascal VOC annotations

## Training Pipeline

### 1. Prepare Your Dataset

#### Organize Images
```
images/
├── train/
│   ├── image1.jpg
│   ├── image1.xml
│   └── ...
└── validation/
    ├── image1.jpg
    ├── image1.xml
    └── ...
```

#### Split Dataset
```bash
python scripts/data_preparation/train_val_test_split.py
```

#### Convert Annotations to CSV
```bash
python scripts/data_preparation/create_csv.py
```

#### Generate TFRecords
```bash
# Training data
python scripts/data_preparation/create_tfrecord.py \
    --csv_input=data/train_labels.csv \
    --image_dir=data/train \
    --labelmap=labelmap.txt \
    --output_path=data/train.tfrecord

# Validation data
python scripts/data_preparation/create_tfrecord.py \
    --csv_input=data/validation_labels.csv \
    --image_dir=data/validation \
    --labelmap=labelmap.txt \
    --output_path=data/validation.tfrecord
```

### 2. Configure Training Pipeline

Edit the pipeline config file (e.g., `configs/ssd_mobilenet_v2_fpnlite_pipeline.config`):

```python
# Key parameters to modify:
- num_classes: 6  # Number of currency denominations
- batch_size: 16
- num_steps: 40000
- fine_tune_checkpoint: "path/to/pretrained/checkpoint"
- train_input_path: "data/train.tfrecord"
- eval_input_path: "data/validation.tfrecord"
- label_map_path: "labelmap.pbtxt"
```

### 3. Train the Model

#### On Google Colab

```bash
# Clone TensorFlow Models repo
!git clone https://github.com/tensorflow/models.git

# Install dependencies
%cd models/research
!protoc object_detection/protos/*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!pip install .

# Start training
!python model_main_tf2.py \
    --pipeline_config_path=../../configs/ssd_mobilenet_v2_fpnlite_pipeline.config \
    --model_dir=../../models/ssd_mobilenet_v2_fpnlite/checkpoint \
    --num_train_steps=40000 \
    --alsologtostderr
```

#### Monitor Training

```bash
# Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir models/ssd_mobilenet_v2_fpnlite/checkpoint
```

### 4. Export Trained Model

```bash
# Export for TFLite
!python models/research/object_detection/export_tflite_graph_tf2.py \
    --pipeline_config_path=configs/ssd_mobilenet_v2_fpnlite_pipeline.config \
    --trained_checkpoint_dir=models/ssd_mobilenet_v2_fpnlite/checkpoint \
    --output_directory=models/ssd_mobilenet_v2_fpnlite/tflite_export
```

### 5. Convert to TensorFlow Lite

```bash
python scripts/training/convert_to_tflite.py
```

This creates `detect.tflite` ready for deployment on Raspberry Pi.

## Training Tips

### Data Augmentation

The pipeline configs include augmentation:
- Random horizontal flip
- Random crop
- SSD random crop (for SSD models)

### Learning Rate

Cosine decay learning rate schedule:
```python
learning_rate_base: 0.08
total_steps: 50000
warmup_learning_rate: 0.026666
warmup_steps: 1000
```

### Batch Size

- Colab/GPU: 16 or 32
- Local CPU: 4 or 8

### Training Steps

- Minimum: 10,000 steps
- Recommended: 40,000 steps
- For best results: 50,000+ steps

## Model Selection

### SSD-MobileNet-v2
- **Best for**: Balance of speed and accuracy
- **Input size**: 300x300
- **Training time**: ~3-4 hours (40K steps on Colab)

### SSD-MobileNet-v2-FPNLite
- **Best for**: Edge devices (Raspberry Pi)
- **Input size**: 320x320  
- **Training time**: ~4-5 hours (40K steps on Colab)
- **Recommended**: ⭐ Best overall choice

### EfficientDet-d0
- **Best for**: Maximum accuracy (on powerful hardware)
- **Input size**: Variable
- **Training time**: ~6-8 hours (40K steps on Colab)

## Troubleshooting

### Out of Memory

```python
# Reduce batch size in config
batch_size: 8  # Instead of 16
```

### Slow Training

- Use Google Colab with GPU runtime
- Reduce image resolution
- Use smaller batch size

### Poor Accuracy

- Increase training steps (50,000+)
- Add more training data
- Check data quality and annotations
- Adjust learning rate

## Next Steps

After training, proceed to:
1. [Evaluation Guide](EVALUATION.md) - Test model performance
2. [Deployment Guide](DEPLOYMENT.md) - Deploy to Raspberry Pi
