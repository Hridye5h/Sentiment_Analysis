# 📊 Visual Sentiment Analysis using Deep Learning

This project implements a **visual sentiment analysis system** that predicts human emotions from facial images and maps them into multiple sentiment scales.

---

## 🚀 Overview

- Dataset: FER2013  
- Models Used:
  - ResNet18 (Transfer Learning)
  - EfficientNet
  - DietNet  
- Framework: PyTorch  

The system:
- Classifies facial emotions (7 classes)
- Converts emotions into sentiment scores
- Supports 3, 5, and 7-level sentiment scales

---

## 🧠 Models Used

### 🔹 ResNet18
- Pretrained model
- Used as baseline
- Frozen layers + custom classifier

### 🔹 EfficientNet
- More efficient and scalable architecture
- Better feature extraction with fewer parameters

### 🔹 DietNet
- Lightweight architecture
- Designed for efficient learning with reduced complexity

---

## 📂 Dataset

- FER2013 (48x48 grayscale images)
- Preprocessing:
  - Resize to 224x224
  - Convert to 3-channel
  - Normalize for pretrained models

---

## ⚙️ Features

- Multi-model comparison (ResNet18, EfficientNet, DietNet)
- Emotion classification (7 classes)
- Multi-scale sentiment mapping (3, 5, 7)
- Training and validation pipeline
- Model evaluation

---

## 📈 Sentiment Mapping

| Emotion   | Sentiment |
|----------|----------|
| Happy    | Positive |
| Neutral  | Neutral  |
| Angry    | Negative |
| Sad      | Negative |
| Surprise | Positive |

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Pandas  
- NumPy  
- Matplotlib  

---
## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/Sentiment_Analysis.git
```

2. Install dependencies:
```bash
pip install torch torchvision pandas matplotlib
```

3. Run the notebook using Jupyter or Google Colab
