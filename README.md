# 📊 Visual Sentiment Analysis using Deep Learning

This project implements a **visual sentiment analysis system** that predicts human emotions from facial images and maps them into multiple sentiment scales (3, 5, and 7 levels).

---

## 🚀 Overview

- Dataset: FER2013  
- Models Used:
  - ConvNeXt-V2  
  - DeiT (Vision Transformer)  
  - EfficientNet-B0  
  - Ensemble Model (Smart Decider)  
- Framework: PyTorch  

The system:
- Classifies facial emotions (7 classes)
- Converts emotions into sentiment scores
- Supports multi-scale sentiment analysis (3, 5, 7)

---

## 🧠 Models Used

### 🔹 ConvNeXt-V2
- Strong feature extraction performance  

### 🔹 DeiT (Vision Transformer)
- Captures global dependencies in images  

### 🔹 EfficientNet-B0
- Efficient and lightweight architecture  

### 🔹 Ensemble Model (Smart Decider)
- Combines predictions from multiple models  
- Achieves highest accuracy  

---

## 📂 Dataset

- FER2013 (48x48 grayscale images)  
- Preprocessing:
  - Resize to 224x224  
  - Convert to 3-channel  
  - Normalize  

---

## ⚙️ Features

- Multi-model comparison  
- Emotion classification (7 classes)  
- Multi-scale sentiment mapping (3, 5, 7)  
- Ensemble learning  

---

## 📊 Results

### 🔹 Model Performance

| Model                      | 7-Scale | 5-Scale | 3-Scale |
|---------------------------|--------|--------|--------|
| ConvNeXt-V2               | 88.59% | 89.73% | 91.69% |
| DeiT (Vision Transformer) | 85.89% | 87.26% | 89.83% |
| EfficientNet-B0           | 82.17% | 83.25% | 86.67% |

---

### 🧠 Ensemble Model

🔥 **Final Accuracy:**
- 7-Scale: **96.12%**
- 5-Scale: **96.38%**
- 3-Scale: **97.10%**

---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/Sentiment_Analysis.git
cd Sentiment_Analysis
pip install torch torchvision pandas matplotlib
```

Open `visual_sentiment_analysis.ipynb` in Jupyter or Google Colab and run all cells.

---

## 📦 Model Weights

Trained weights are not included due to size limitations.  
You can train the models using the provided notebook.
