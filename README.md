# Multi-Label Classification of Thoracic Diseases using DenseNet-121

A Deep Learning project for the automatic detection and multi-label classification of 14 distinct chest X-ray pathologies using the **NIH Chest X-ray** dataset. The pipeline employs **DenseNet-121** via transfer learning, fine-tuning strategies, and a custom **Weighted Binary Cross-Entropy** loss function to address class imbalance.

---

## Project Overview

Since a single chest X-ray can present multiple overlapping conditions, this problem is formulated as a **multi-label classification** task across **14 target pathologies**:

> Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax.

---

## Key Features & Methodology

- **Architecture:** DenseNet-121 pretrained on ImageNet.
- **Data Augmentation:** Real-time transformations (`ImageDataGenerator`) including rotation, zoom, spatial shifts, and horizontal flips applied to the training set.
- **Class Imbalance Mitigation:** Custom **Weighted Binary Cross-Entropy (WBCE)** loss function penalizing misclassifications on rare diseases based on inverse class frequency.
- **Two-Stage Training:**
  1. *Frozen Backbone:* Initial training of the classification head (GAP + Dense Layer with Sigmoid activation).
  2. *Fine-Tuning:* Unfreezing the last 60 layers with a lower learning rate ($10^{-5}$) for fine feature adaptation.
- **Optimization Callbacks:** EarlyStopping, ModelCheckpoint (tracking validation AUC), and ReduceLROnPlateau.

---

## Key Results & Metrics

Evaluation on the unseen test set demonstrates robust diagnostic performance across both frequent and rare pathologies:

- **Micro-Average ROC-AUC:** `0.773`
- **Macro-Average ROC-AUC:** `0.770`
- Near-identical Micro and Macro AUC scores confirm that the custom loss function successfully prevented bias toward majority classes.

---

## Project Structure

```text
├── Data_Entry_2017.csv             # Dataset metadata
├── multi-label-chest-x-ray.ipynb   # Model training & pipeline notebook
├── evaluation.ipynb                # Results analysis & visualization notebook
├── best_model.h5                   # Saved Keras model checkpoint
└── results/                        # Visualizations & ROC-AUC curves
```

---

## Tech Stack

- **Framework:** TensorFlow / Keras
- **Data & Math:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
