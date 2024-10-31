# Diarization Pipeline Script

## Overview

This script processes audio files in three main stages:

1. **Conversion to WAV format** – Converts input files to WAV, which is CPU-bound.
2. **Background Noise Reduction** – Reduces background noise, requiring GPU processing.
3. **Speaker Diarization** – Analyzes audio for speaker segmentation, also GPU-bound.

The pipeline is designed for efficient concurrent processing, utilizing both CPU and GPU resources to improve performance.

## Why More CPU Workers Than GPU Workers?

- **CPU-bound tasks** (like file conversion) benefit from parallel processing, so `max_workers` is set to 4 by default, allowing up to 4 concurrent CPU threads.
- **GPU-bound tasks** (background noise reduction and diarization) are limited by `GPU_SEMAPHORE`, set to a maximum of 2. This is because each GPU-bound task requires approximately 4GB of VRAM, making two concurrent tasks optimal for GPUs with 8GB VRAM. If your GPU has less VRAM, consider reducing the `GPU_SEMAPHORE` to 1 to avoid overloading.

## Requirements

1. **Install dependencies:**
   pip3 install -r requirements.txt

2. **CUDA:** Install the CUDA library on your host machine to enable GPU support.
   - [CUDA Download](https://developer.nvidia.com/cuda-downloads)

3. **PyTorch with CUDA support:** Install PyTorch with CUDA enabled for GPU acceleration.
   - [PyTorch Installation Guide](https://pytorch.org/)

## Usage

1. Place your input audio files in the desired directory (e.g., `"Top Gear Specials"`).
2. Run the script with the following command:
   python diarization_pipeline.py
