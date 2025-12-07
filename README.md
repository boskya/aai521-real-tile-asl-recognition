# Real-Time ASL Alphabet Recognition

**University of San Diego - AAI-521 Computer Vision**  
**Author:** Bosky Atlani  
**Project:** Real-Time American Sign Language Alphabet Recognition with Cross-Dataset Evaluation

---

## Overview

This project implements a deep learning system for recognizing American Sign Language (ASL) alphabet gestures using transfer learning with EfficientNet-B0. The system includes model training, cross-dataset evaluation, and a real-time inference web application built with Streamlit.

**Key Findings:**
- Achieved 92.5% validation accuracy on in-distribution test data
- Demonstrated significant generalization gap: 28.5% on cross-dataset evaluation, 60.7% on real-world webcam testing
- Identified incompatible gesture definitions across different ASL datasets
- Highlighted the critical importance of training data diversity for robust real-world deployment

---

## Repository Structure

```
aai521-real-time-asl-recognition/
├── asl.ipynb          # Model training in Google Colab
├── run_asl.py                       # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

Data, model and checkpoint results are saved on google drive. Please contact author for assitance with this. 

---

## Features

### 1. Model Training (`asl.ipynb`)
- **Architecture:** EfficientNet-B0 with transfer learning
- **Dataset:** Kaggle ASL Alphabet (87,000 images, 29 classes)
- **Two Training Strategies:**
  - Model 1: Baseline (frozen base, no augmentation)
  - Model 2: Enhanced (data augmentation + fine-tuning)
- **Data Augmentation:** Rotation (±15°), zoom (±15%), brightness variation (±30%), spatial shifts
- **Two-Stage Training:** Classifier-only → Full fine-tuning

### 2. Cross-Dataset Evaluation (`cross_dataset_evaluation.ipynb`)
- Tests model generalization on Ayuraj ASL dataset (1,815 images)
- Generates confusion matrices and per-class accuracy analysis
- Identifies gesture definition incompatibilities across datasets

### 3. Real-Time Inference Application (`run_asl.py`)
- **Web Interface:** Streamlit-based interactive application
- **Hand Detection:** MediaPipe Hands with 21 landmark detection
- **Preprocessing Pipeline:**
  - Bounding box extraction with percentage-based padding
  - Optional background segmentation
  - EfficientNet preprocessing
- **Testing Protocol:** Systematic data collection with CSV export
- **Results Dashboard:** Real-time accuracy tracking per letter

---

## Installation

### Prerequisites
- Python 3.10 or 3.11 (Python 3.9 has compatibility issues)
- GPU optional (works on CPU, Metal GPU on Mac)

### Setup
```bash
# Clone repository
git clone https://github.com/boskya/aai521-real-time-asl-recognition.git
cd aai521-real-time-asl-recognition

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies
```txt
tensorflow==2.13.0          # Or tensorflow-macos for Apple Silicon
streamlit==1.28.0
opencv-python-headless==4.8.1.78
mediapipe==0.10.8
numpy<2.0                   # TensorFlow compatibility
pillow==10.0.0
pandas==2.0.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
```

---

## Usage

### Training Models (Google Colab)

1. Open `asl_training.ipynb` in Google Colab
2. Mount Google Drive for model/checkpoint storage
3. Download Kaggle ASL Alphabet dataset
4. Run all cells to train Model 1 and Model 2
5. Models saved to Google Drive


### Real-Time Inference Application (Local)
```bash
# Ensure model file is in project directory
# Download from Google Drive: asl_final_best.keras or asl_model_best.keras

# Run Streamlit app
streamlit run run_asl.py

# Application opens in browser at http://localhost:8501
```

**Application Features:**
- **Tab 1:** Upload/capture images for immediate prediction
- **Tab 2:** Structured testing protocol (record correct/incorrect)
- **Tab 3:** Results aggregation with per-letter accuracy
- **Export:** Download results as CSV

---


## Technical Details

### Model Architecture
```
Input (224×224×3)
    ↓
EfficientNet-B0 (pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(29, Softmax)
```

### Preprocessing Pipeline
1. **Hand Detection:** MediaPipe Hands (21 landmarks)
2. **Bounding Box:** Percentage-based padding (50-100% of hand size)
3. **Background Segmentation:** Convex hull mask with white background replacement
4. **Resize:** 224×224 bilinear interpolation
5. **Color Conversion:** BGR → RGB
6. **Normalization:** EfficientNet `preprocess_input` (ImageNet statistics)

### Classes (29 total)
A-Z, SPACE, DELETE, NOTHING

---

## Datasets

### Kaggle ASL Alphabet (Training)
- **Size:** 87,000 images (200×200 RGB)
- **Classes:** 29
- **Characteristics:** Uniform backgrounds, consistent lighting, centered composition
- **Split:** 80% train, 20% validation
- **Source:** [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

### Ayuraj ASL Dataset (Cross-Dataset Testing)
- **Size:** 1,815 images
- **Classes:** 26 (alphabet only)
- **Characteristics:** Varied backgrounds, different hand orientations, inconsistent lighting
- **Source:** [Kaggle Ayuraj ASL Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

**Note:** Datasets not included in repository due to size. Download separately from Kaggle.

---

## Known Issues & Limitations

### Technical Issues
- **Python 3.9 Compatibility:** Use Python 3.10+ to avoid TensorFlow/NumPy conflicts

### Performance Limitations
- **Overfitting to Training Distribution:** Model learned dataset-specific patterns (backgrounds, lighting)
- **Incompatible Gesture Definitions:** 10+ letters use different hand configurations in Ayuraj vs. Kaggle
- **Webcam Sensitivity:** Performance degrades with cluttered backgrounds, poor lighting, non-frontal orientations
- **Bounding Box Cropping:** Tight crops may cut off fingers; requires careful padding adjustment

### Deployment Considerations
- Real-world accuracy (60.7%) significantly below validation accuracy (92.5%)
- Requires training conditions to be matched for reliable predictions
- Background segmentation helps but doesn't solve fundamental domain shift

---

## License

This project is for educational purposes as part of the AAI-521 Computer Vision course at the University of San Diego.

---

## Acknowledgments

- **Dataset:** Kaggle ASL Alphabet dataset by @grassknoted
- **Cross-Dataset:** Ayuraj ASL Dataset by @ayuraj
 **AI Development Assistant:** Claude (Anthropic) for code generation and debugging support
