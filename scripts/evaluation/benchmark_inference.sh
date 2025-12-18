#!/bin/bash

# Benchmark Inference Speed for TensorFlow Lite Models
# Run this script on Raspberry Pi to measure inference times

echo "================================"
echo "TFLite Model Benchmark Tool"
echo "================================"
echo ""

# Check if benchmark tool exists
if [ ! -f "linux_arm_benchmark_model_plus_flex" ]; then
    echo "Downloading Google's benchmark tool..."
    wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_plus_flex
    chmod +x linux_arm_benchmark_model_plus_flex
    echo "Download complete!"
    echo ""
fi

# Configuration
NUM_THREADS=4
NUM_RUNS=50

# Function to benchmark a model
benchmark_model() {
    local model_name=$1
    local model_path=$2
    
    echo "Benchmarking: $model_name"
    echo "Model path: $model_path"
    echo "Threads: $NUM_THREADS, Runs: $NUM_RUNS"
    echo "---"
    
    ./linux_arm_benchmark_model_plus_flex \
        --graph=$model_path \
        --num_threads=$NUM_THREADS \
        --num_runs=$NUM_RUNS \
        --warmup_runs=10
    
    echo ""
    echo "================================"
    echo ""
}

# Benchmark all models
if [ -f "models/ssd_mobilenet_v2/detect.tflite" ]; then
    benchmark_model "SSD-MobileNet-v2" "models/ssd_mobilenet_v2/detect.tflite"
fi

if [ -f "models/ssd_mobilenet_v2_fpnlite/detect.tflite" ]; then
    benchmark_model "SSD-MobileNet-v2-FPNLite" "models/ssd_mobilenet_v2_fpnlite/detect.tflite"
fi

if [ -f "models/efficientdet_d0/detect.tflite" ]; then
    benchmark_model "EfficientDet-d0" "models/efficientdet_d0/detect.tflite"
fi

echo "Benchmark complete!"
echo ""
echo "Results saved to benchmark_results.txt"
