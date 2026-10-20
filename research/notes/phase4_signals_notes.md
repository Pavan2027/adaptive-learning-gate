# Phase 4 — Learning Signals Formulation & Analysis

**Date:** 2026-10-21  
**Status:** Completed & Tested  

---

## 1. Signal Mathematical Formulations

All signals are mathematically mapped to normalized scalar values in $[0.0, 1.0]$.

### 1.1 Predictive Uncertainty ($U \in [0.0, 1.0]$)
Derived from normalized Shannon Entropy of the model's softmax predictive distribution:
$$U(x) = -\frac{1}{\log_2(C)} \sum_{c=1}^C p_c \log_2(p_c + \epsilon)$$
- $U = 0.0$: Minimum uncertainty (model predicts a single class with 100% confidence).
- $U = 1.0$: Maximum uncertainty (uniform distribution over all $C$ classes).

### 1.2 Semantic Novelty ($N \in [0.0, 1.0]$)
Measures the distance from the sample's penultimate feature embedding $z \in \mathbb{R}^D$ to the nearest known class prototype centroid $\mu_c$:
$$N(z) = \min_{c} \frac{1 - \cos(z, \mu_c)}{2}$$
- $N = 0.0$: Representation is identical to an established prototype (redundant observation).
- $N = 1.0$: Representation is orthogonal or opposite to all established prototypes.
- **Memory Complexity:** Bounded $\mathcal{O}(C \cdot D)$ memory footprint (~10 KB for CIFAR-10 with $D=256$). Class centroids update online via Exponential Moving Average (EMA).

### 1.3 Sample Reliability ($R \in [0.0, 1.0]$)
Composes input feature sanity with predictive label consistency:
$$R(x, \tilde{y}) = R_{\text{feat}}(x) \times R_{\text{label}}(x, \tilde{y})$$
where:
1. $R_{\text{feat}}(x)$: Evaluates finite values and flags abnormal spatial pixel variance ($\sigma > 2.5$).
2. $R_{\text{label}}(x, \tilde{y}) = \frac{p(\tilde{y} \mid x)}{\max_c p(c \mid x) + \epsilon}$:
   - When the model strongly predicts class $A$, but the observed label claims class $B$, $R_{\text{label}} \to 0$.
   - When observed label matches top prediction, $R_{\text{label}} = 1.0$.

---

## 2. Computational Overhead
- **Forward-Only:** Signals are computed in evaluation mode (`torch.no_grad()`) in a single forward pass without backpropagation.
- **Zero Label Leakage:** The signals evaluate only $(x, \tilde{y})$ and never access ground truth labels or synthetic noise indicators.
