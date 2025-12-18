# Raspberry Pi Deployment Guide

Complete guide for deploying your trained TensorFlow Lite model on Raspberry Pi 4.

## Hardware Setup

### Required Components
- Raspberry Pi 4 (4GB+ RAM recommended)
- MicroSD card (32GB+ recommended)
- USB Webcam or Raspberry Pi Camera Module
- Power supply (5V 3A)
- (Optional) Cooling fan or heatsink

### Recommended Accessories
- Case with fan
- 64GB+ microSD card for datasets
- Coral USB Accelerator (for faster inference)

## Software Installation

### 1. Install Raspberry Pi OS

Download and flash Raspberry Pi OS (64-bit recommended):
```bash
# Using Raspberry Pi Imager
# Select: Raspberry Pi OS (64-bit)
```

### 2. Initial Setup

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.9
sudo apt-get install python3.9 python3.9-venv python3-pip -y
```

### 3. Install Dependencies

```bash
# System libraries
sudo apt-get install -y libatlas-base-dev libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install -y libopenblas-dev liblapack-dev gfortran
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
sudo apt-get install -y libopencv-dev python3-opencv

# Create project directory
mkdir ~/currency-detection
cd ~/currency-detection

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install TensorFlow Lite Runtime
pip3 install --upgrade pip
pip3 install tflite-runtime==2.8.0

# Install other dependencies
pip3 install opencv-python numpy pillow
```

### 4. Transfer Model Files

From your development machine:
```bash
# Using SCP
scp -r models/ pi@raspberrypi.local:~/currency-detection/
scp scripts/detection/TFLite_detection_webcam.py pi@raspberrypi.local:~/currency-detection/
scp data/labelmap.txt pi@raspberrypi.local:~/currency-detection/
```

Or clone from GitHub:
```bash
cd ~/currency-detection
git clone https://github.com/yourusername/usd-currency-detection.git
cd usd-currency-detection
```

## Running Detection

### Basic Usage

```bash
cd ~/currency-detection
source venv/bin/activate

python TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite \
    --graph detect.tflite \
    --labels labelmap.txt
```

### Configuration Options

```bash
# Adjust detection threshold
--threshold 0.6  # Default: 0.5

# Change camera resolution
--resolution 640x480  # Lower for better FPS

# Use different camera
--camera 1  # If you have multiple cameras
```

### Camera Setup

**For USB Webcam:**
```bash
# Test camera
ls /dev/video*
# Should show /dev/video0

# View camera feed
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

**For Pi Camera:**
```bash
# Enable camera interface
sudo raspi-config
# Interface Options → Camera → Enable

# Test Pi Camera
libcamera-hello
```

## Performance Optimization

### 1. Overclock (Optional, be careful!)

Edit `/boot/config.txt`:
```bash
sudo nano /boot/config.txt

# Add these lines (Pi 4):
over_voltage=6
arm_freq=2000
gpu_freq=750
```

**Warning:** Use adequate cooling and monitor temperatures!

### 2. CPU Optimization

```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check CPU frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### 3. Memory Split

Allocate more RAM to CPU:
```bash
sudo raspi-config
# Performance Options → GPU Memory → Set to 128
```

### 4. Disable Unnecessary Services

```bash
# Disable Bluetooth
sudo systemctl disable bluetooth

# Disable WiFi (if using Ethernet)
sudo systemctl disable wpa_supplicant
```

## Using Coral Edge TPU

For significantly faster inference (~10x speedup):

### 1. Install Edge TPU Runtime

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std python3-pycoral
```

### 2. Convert Model for Edge TPU

On development machine:
```bash
edgetpu_compiler detect.tflite
# Creates: detect_edgetpu.tflite
```

### 3. Run with Edge TPU

```bash
python TFLite_detection_webcam.py \
    --modeldir models/ssd_mobilenet_v2_fpnlite \
    --graph detect_edgetpu.tflite \
    --labels labelmap.txt \
    --edgetpu
```

## Autostart on Boot

### Create Systemd Service

```bash
sudo nano /etc/systemd/system/currency-detection.service
```

Add:
```ini
[Unit]
Description=Currency Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/currency-detection
ExecStart=/home/pi/currency-detection/venv/bin/python TFLite_detection_webcam.py --modeldir models/ssd_mobilenet_v2_fpnlite
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable currency-detection
sudo systemctl start currency-detection

# Check status
sudo systemctl status currency-detection
```

## Troubleshooting

### Camera Not Working

```bash
# Check camera permissions
groups $USER
# Should include 'video'

# Add user to video group
sudo usermod -a -G video $USER

# Reboot
sudo reboot
```

### Low FPS

**Solutions:**
1. Lower camera resolution: `--resolution 640x480`
2. Use recommended model (FPNLite)
3. Close other applications
4. Consider Coral Edge TPU
5. Overclock (with cooling!)

### Out of Memory

```bash
# Check memory
free -h

# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Model Loading Errors

```bash
# Verify model file
ls -lh models/ssd_mobilenet_v2_fpnlite/detect.tflite

# Check TFLite installation
python -c "import tflite_runtime; print(tflite_runtime.__version__)"
```

## Headless Mode (No Display)

For running without monitor:

```bash
# Add to script
export DISPLAY=:0

# Or modify detection script
# Add --no_display flag (requires code modification)
```

## Remote Access

### VNC Server

```bash
# Enable VNC
sudo raspi-config
# Interface Options → VNC → Enable

# Connect from computer
# Download RealVNC Viewer
# Connect to: raspberrypi.local
```

### SSH

```bash
# From your computer
ssh pi@raspberrypi.local

# Run detection
cd ~/currency-detection
source venv/bin/activate
python TFLite_detection_webcam.py --modeldir models/ssd_mobilenet_v2_fpnlite
```

## Performance Benchmarks

### Expected Performance (Raspberry Pi 4, 4GB)

| Model | Inference Time | FPS | Usability |
|-------|---------------|-----|-----------|
| SSD-MobileNet-v2 | 170 ms | 2.7 | ✅ Good |
| SSD-MobileNet-v2-FPNLite | 134 ms | 2.0 | ✅ Best |
| EfficientDet-d0 | 1284 ms | 0.5 | ⚠️ Too slow |

### With Coral Edge TPU

| Model | Inference Time | FPS |
|-------|---------------|-----|
| SSD-MobileNet-v2-FPNLite | ~15 ms | ~30 | 

## Power Consumption

Typical power draw:
- Idle: 2.5W
- Running detection: 5-7W
- With Coral TPU: 7-9W

Use quality 5V 3A power supply!

## Next Steps

1. Test detection performance
2. Fine-tune confidence threshold
3. Add audio feedback for accessibility
4. Build custom enclosure
5. Deploy in real-world application
