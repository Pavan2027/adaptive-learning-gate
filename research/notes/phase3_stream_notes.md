# Phase 3 — Stream Simulator & Controlled Perturbations

**Date:** 2026-10-20  
**Status:** Completed & Tested  

---

## 1. Stream Design & Mathematical Formulation

The continual learning data stream $\mathcal{S} = \{(x_t, \tilde{y}_t)\}_{t=1}^T$ delivers samples sequentially in discrete time steps $t$.

### Perturbation Types Supported

1. **Clean Samples:** Unmodified image $x_t$ and true label $y_t$.
2. **Symmetric Label Noise ($\eta_{\text{label}}$):**
   $$\tilde{y}_t = \begin{cases} y_t & \text{with probability } 1 - \eta_{\text{label}} \\ y' \sim \text{Uniform}(\{0, \dots, C-1\} \setminus \{y_t\}) & \text{with probability } \eta_{\text{label}} \end{cases}$$
3. **Feature Corruption ($\eta_{\text{feat}}$):**
   - **Gaussian Noise:** $\tilde{x}_t = x_t + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.
   - **Cutout Masking:** Rectangular region masked to zero to simulate occlusion.
   - **Blur:** Local spatial box smoothing.
4. **Redundancy ($\eta_{\text{redundant}}$):** Repeated sampling of recent observations to test uninformative update skipping.
5. **Distribution Shift:** Segment-based non-stationary shifts (e.g. changing class priors across segments $k \in \{1, \dots, K\}$).

---

## 2. Strict Fairness & Zero-Leakage Guarantee

To prevent synthetic leakage during training and gate evaluation:
- The `StreamSample.training_tuple()` method returns **strictly `(image, observed_label)`**.
- Ground-truth attributes (`true_label`, `is_noisy_label`, `is_corrupted_feature`, `is_redundant`) are quarantined inside `SampleMetadata` and used **only** for post-hoc paper evaluation (e.g., computing the gate's precision in filtering corrupted samples).
- Stream manifests are serialized to JSON (`stream_manifest.json`) so all algorithms face the **exact same stream sequence**.
