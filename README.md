# Real-Time ASL Alphabet Recognition

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

Real-time American Sign Language (ASL) alphabet recognition system using deep learning. 
Recognizes 29 ASL gestures (A-Z + SPACE, DELETE, NOTHING) from webcam video at 20+ FPS.

**Course:** AAI-521 Computer Vision  
**Institution:** University of San Diego  
**Author:** Bosky Atlani, Team 9
**Date:** 2025

## 🎯 Key Features

- EfficientNet-B0 transfer learning model
- MediaPipe hand detection and isolation
- ⚡ Real-time inference (20+ FPS)
- 🎥 Live webcam demonstration


## 📊 Results

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data
```bash
# Download datasets from Kaggle
bash scripts/download_data.sh

# Or manually download:
# 1. https://www.kaggle.com/datasets/grassknoted/asl-alphabet
# Place in data/raw/
```


## 📁 Project Structure

## 🎓 Methodology

### Dataset


### Model Architecture

### Training

### Real-Time Pipeline
1. Capture webcam frame (30 FPS)
2. MediaPipe hand detection
3. Crop to hand bounding box
4. Resize to 224×224 and normalize
5. EfficientNet-B0 inference
6. Majority voting (5-frame buffer)
7. Display prediction

## 📈 Performance

### Offline Evaluation

### Real-World Testing

### Robustness Analysis

