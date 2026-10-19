# Phase 2 — Baseline Experiment Notes

**Date:** 2026-10-20  
**Status:** Completed & Verified  

---

## 1. Experimental Setup

- **Model:** `CompactCNN` (4-layer Convolutional Neural Network + BatchNorm + Dropout + Adaptive Pooling)
- **Trainable Parameters:** 620,842
- **Dataset:** CIFAR-10 (45,000 Train, 5,000 Validation, 10,000 Held-Out Test)
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Acceleration:** PyTorch AMP (`torch.amp.autocast('cuda')` with FP16), `torch.backends.cudnn.benchmark = True`
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **Batch Size:** 256
- **Epochs:** 5
- **Seed:** 42

---

## 2. Quantitative Results

| Metric | Measured Value |
| :--- | :--- |
| **Best Validation Accuracy** | **71.40%** (Epoch 5) |
| **Held-Out Test Accuracy** | **73.35%** |
| **Held-Out Test Macro-F1** | **73.29%** |
| **Held-Out Test Loss** | **0.7603** |
| **Total Steps** | 880 |
| **Total Wall-Clock Time** | **54.26 seconds** (~10.8s / epoch) |

### Per-Class Test Accuracies (CIFAR-10 Classes 0–9)
- Class 0 (Airplane): 72.7%
- Class 1 (Automobile): 81.4%
- Class 2 (Bird): 51.2%
- Class 3 (Cat): 62.7%
- Class 4 (Deer): 79.9%
- Class 5 (Dog): 62.1%
- Class 6 (Frog): 90.0%
- Class 7 (Horse): 60.7%
- Class 8 (Ship): 83.2%
- Class 9 (Truck): 89.6%

---

## 3. Key Findings for the Paper
1. **Strong Baseline Anchor:** The 73.35% accuracy achieved in just 54 seconds provides an optimal, reproducible reference point for evaluating selective updates in continual stream settings.
2. **Class Confusion Matches Literature:** Classes with high semantic ambiguity (Bird, Cat, Dog, Horse) show expected lower accuracies (~51–62%), providing ideal scenarios where predictive uncertainty and reliability signals can help the Adaptive Learning Gate filter ambiguous or noisy samples.
