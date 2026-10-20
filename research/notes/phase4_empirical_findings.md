# Phase 4 — Empirical Signal Analysis & Paper Insights

**Date:** 2026-10-21  
**Status:** Completed & Empirically Validated  

---

## 1. Empirical Signal Distribution on CIFAR-10 Stream (N = 2,000)

Using the trained `CompactCNN` baseline model evaluated on the perturbed CIFAR-10 stream, we observed the following distribution of signals:

| Sample Condition | Predictive Uncertainty (Shannon Entropy) | Sample Reliability Score | Semantic Novelty (Cosine Distance) |
| :--- | :--- | :--- | :--- |
| **Clean Samples** | $0.2936 \pm 0.2194$ | **$0.8514 \pm 0.3024$** | $0.1277 \pm 0.0459$ |
| **Noisy Labels** | $0.3043 \pm 0.2261$ | **$0.0734 \pm 0.1986$** | — |
| **Corrupted Features** | $0.4174 \pm 0.2128$ | $0.5177 \pm 0.4374$ | $0.1510 \pm 0.0419$ |

---

## 2. Core Scientific Finding for the Paper

### Critical Insight: "Why Uncertainty Alone Fails to Detect Label Noise"
1. **The Fallacy of Uncertainty-Only Selection (Active Learning):**
   - Clean samples exhibit an average uncertainty of **0.2936**.
   - Noisy-label samples exhibit an average uncertainty of **0.3043**.
   - **Difference:** Only $+0.0107$. Standard predictive uncertainty (entropy / margin) **cannot distinguish between clean and mislabeled samples**. Under traditional active learning or uncertainty sampling, noisy labels would be readily admitted for model updates, severely corrupting model parameters.

2. **The Discriminative Power of Our Reliability Signal:**
   - Clean sample reliability: **$0.8514$**
   - Noisy-label sample reliability: **$0.0734$**
   - **Reliability Separation Gap: $+0.7780$ (11.6× separation factor).**
   - Our predictive label consistency signal detects conflict between model representation structure and the observed label without requiring access to oracle ground-truth flags.

3. **Feature Corruption Detection:**
   - Feature corruptions (Gaussian noise, cutout, blur) significantly elevate uncertainty ($0.4174$ vs $0.2936$) and degrade reliability ($0.5177$ vs $0.8514$), enabling dual-signal filtering.
