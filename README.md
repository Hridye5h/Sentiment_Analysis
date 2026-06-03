# Visual Sentiment Analysis — Facial Emotion Recognition (RAF-DB)

Facial emotion recognition on **RAF-DB**, mapped onto **3-, 5-, and 7-level sentiment scales**. The project
fine-tunes a **DDAMFN** model and builds a two-model **stacking ensemble** (DDAMFN + ConvNeXt-V2), reaching
**91.6% overall accuracy** on the official test set — on par with published state of the art.

---

## Overview
- **Task:** 7-class facial emotion recognition — *Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral*.
- **Sentiment scales:** 5-level, and 3-level valence (positive / neutral / negative).
- **Framework:** PyTorch. **Trained on:** Kaggle (T4 / P100 GPU).

## Dataset — RAF-DB (Real-world Affective Faces Database)
- **RAF-DB basic** subset: **12,271 training** / **3,068 test** images across 7 basic emotions.
- Aligned RGB faces. The dataset is **imbalanced** — Happiness dominates while Fear and Disgust are rare,
  which is reflected in the mean-class (balanced) accuracy below.

## Results

### Single model — DDAMFN
| Metric | Score |
| --- | --- |
| **7-class overall accuracy** | **91.07%** |
| 7-class mean-class (balanced) | 84.64% |
| 5-scale accuracy | 92.11% |
| 3-scale (valence) accuracy | 92.89% |
| Trainable parameters | 4.19 M |

Per-class (test set):

| Emotion | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Surprise | 0.895 | 0.881 | 0.888 | 329 |
| Fear | 0.769 | 0.676 | 0.719 | 74 |
| Disgust | 0.792 | 0.737 | 0.764 | 160 |
| Happiness | 0.965 | 0.970 | 0.968 | 1185 |
| Sadness | 0.894 | 0.881 | 0.887 | 478 |
| Anger | 0.914 | 0.858 | 0.885 | 162 |
| Neutral | 0.875 | 0.921 | 0.898 | 680 |

### Stacking ensemble (DDAMFN + ConvNeXt-V2)
Ablation on the test set:

| Model | Overall | Mean-class |
| --- | --- | --- |
| DDAMFN | 91.07% | 84.64% |
| ConvNeXt-V2 | 87.29% | 78.10% |
| Soft-vote average | 91.59% | 84.95% |
| **Stacking (Logistic Regression)** | **91.62%** | 83.69% |

Ensemble sentiment scales: **3-scale 93.61%**, **5-scale 92.76%**. The soft-vote average gives the best
balance of overall and mean-class accuracy, improving on either base model.

### State-of-the-art context (RAF-DB, 7-class overall accuracy)
| Method | Overall acc | Year |
| --- | --- | --- |
| Ig3D | 94.0% | 2024 |
| POSTER++ | 92.21% | 2023 |
| **This project (ensemble)** | **91.62%** | — |
| DDAMFN (paper) | 91.35% | 2023 |

## Method
- **DDAMFN** (Dual-Direction Attention Mixed Feature Network): a MixedFeatureNet backbone pretrained on
  **MS-Celeb-1M**, with a dual-direction attention head, trained using **SAM** (Sharpness-Aware Minimization)
  and an attention-diversity loss at 112×112 (only 4.19 M parameters).
- **ConvNeXt-V2** (tiny): ImageNet-pretrained, fine-tuned with AdamW, label smoothing, and a cosine schedule at 224×224.
- **Ensemble:** a Logistic Regression meta-learner over the two models' softmax probabilities (14 features).

## Evaluation protocol
- A stratified **validation** split is carved from the training data; the **3,068 test images are evaluated once**.
- Both base models train on **train**; the meta-learner is fit on **validation**; **test** is held out for final reporting.
- Reports **overall** and **mean-class (balanced)** accuracy, with one consistent 3-/5-scale mapping.

## Reproduce
On Kaggle: add the **`raf-db-emotion-classification-challenge`** dataset, enable **GPU** and **Internet**, then Run All.

| Notebook | Description |
| --- | --- |
| `sota-facial-recognition.ipynb` | DDAMFN fine-tuning + evaluation (single model). |
| `facial-emotion-recognition.ipynb` | DDAMFN + ConvNeXt-V2 stacking ensemble with ablation. |

## Limitations
- **Class imbalance:** Fear (recall 0.68) and Disgust (0.74) are the weakest classes due to few samples;
  mean-class accuracy trails overall accuracy by ~6 points as a result.
- **Sentiment scales:** the 3-/5-scale numbers are higher than 7-class because grouping emotions yields coarser classes.
- All figures are on the standard RAF-DB test split and are not directly comparable to other datasets.

## References
- **DDAMFN** — S. Zhang et al., *A Dual-Direction Attention Mixed Feature Network for Facial Expression
  Recognition*, Electronics 12(17), 2023. <https://github.com/SainingZhang/DDAMFN>
- **POSTER++** — J. Mao et al., *POSTER V2: A Simpler and Stronger Facial Expression Recognition Network*, 2023.
- **RAF-DB** — S. Li, W. Deng, *Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression
  Recognition in the Wild*, CVPR 2017.
