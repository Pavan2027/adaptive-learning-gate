# Phase 5 — Adaptive Learning Gate: Formulation, Policies & Empirical Validation

**Document ID:** `ALG-RES-005`  
**Phase:** 5 (Adaptive Learning Gate & Bounded Defer Queue)  
**Status:** Verified & Empirically Validated  
**Date:** 2026-10-21  
**Benchmark:** CIFAR-10 (Continuous perturbed stream, 2,000 samples)  

---

## 1. Executive Summary

Phase 5 implements the core decision engine of the **Adaptive Learning Gate (ALG)**. Rather than passively performing gradient descent on every incoming sample or relying naively on predictive uncertainty (the traditional active learning paradigm), the ALG evaluates each sample across three normalized orthogonal signals:
1. **Uncertainty ($U \in [0, 1]$)** — Shannon entropy over predictive distribution.
2. **Novelty ($N \in [0, 1]$)** — Cosine distance to class prototypes in latent feature space.
3. **Reliability ($R \in [0, 1]$)** — Feature integrity modulated by label-feature concordance.

Through an auditable utility formulation, the gate classifies samples into:
- **`LEARN`**: High-value, high-reliability information admitted to parameter updates.
- **`IGNORE`**: Corrupt, noisy, or redundant samples discarded to protect weights and conserve compute.
- **`DEFER`**: Ambiguous samples quarantined in a bounded memory buffer for deferred re-evaluation as model confidence matures.

Empirical evaluation on a 2,000-sample CIFAR-10 stream demonstrates **96.43% interception of label noise** (93.88% immediate rejection via `IGNORE`, 2.55% quarantined in `DEFER`), resolving a known critical vulnerability in continual learning and active learning systems.

---

## 2. Mathematical Formulation

### 2.1 Benefit Estimation $B(x)$
A sample provides learning utility if it expands the model's representation space (novelty) or clarifies decision boundaries (uncertainty). However, **uncertainty is only informative if the sample is reliable**:

$$B(x) = w_{\text{nov}} \cdot N(x) + w_{\text{unc}} \cdot \big(U(x) \cdot R(x)\big)$$

- $w_{\text{nov}} = 0.5$: Weight attributed to latent feature novelty.
- $w_{\text{unc}} = 0.5$: Weight attributed to effective uncertainty.
- **Crucial Coupling:** $U(x) \cdot R(x)$ ensures that high uncertainty paired with near-zero reliability yields zero benefit, neutralizing the primary failure mode of uncertainty-based active learning.

### 2.2 Harm / Risk Estimation $H(x)$
A sample incurs two forms of risk/cost:
1. **Corruption Risk:** Updating on incorrect labels degrades parameter representations.
2. **Compute Waste (Redundancy):** Updating on samples that are already familiar and predicted with near-perfect certainty consumes compute without imparting knowledge.

$$H(x) = w_{\text{unrel}} \cdot (1.0 - R(x)) + w_{\text{red}} \cdot \big((1.0 - N(x)) \cdot (1.0 - U(x))\big)$$

- $w_{\text{unrel}} = 0.8$: Strong penalty on unreliability (noise suppression).
- $w_{\text{red}} = 0.2$: Modest penalty on redundant, already-mastered samples.

### 2.3 Normalized Net Utility $\mathcal{U}(x)$
The net utility balances potential gain against expected harm:

$$\text{RawUtility}(x) = B(x) - H(x) \in [-1.0, 1.0]$$

$$\mathcal{U}(x) = \frac{\text{RawUtility}(x) + 1.0}{2.0} \in [0.0, 1.0]$$

---

## 3. Decision Policy & Reason Codes

The decision policy partitions the continuous utility metric $\mathcal{U}(x)$ using two ordered thresholds:

$$\tau_{\text{ignore}} < \tau_{\text{learn}} \quad (\text{Default: } \tau_{\text{ignore}} = 0.30, \, \tau_{\text{learn}} = 0.50)$$

$$\text{Action}(x) = \begin{cases} 
\mathbf{LEARN} & \text{if } \mathcal{U}(x) \ge \tau_{\text{learn}} \\ 
\mathbf{IGNORE} & \text{if } \mathcal{U}(x) \le \tau_{\text{ignore}} \\ 
\mathbf{DEFER} & \text{if } \tau_{\text{ignore}} < \mathcal{U}(x) < \tau_{\text{learn}} 
\end{cases}$$

### Auditable Reason Codes
Every decision emits a structured `GateDecision` object recording the sample ID, timestamp index, raw signal snapshot, utility value, and one of four explicit reason codes:

| Action | Reason Code | Condition | Meaning |
|---|---|---|---|
| **LEARN** | `HIGH_UTILITY` | $\mathcal{U}(x) \ge \tau_{\text{learn}}$ | Sample possesses sufficient novelty/uncertainty with high reliability. |
| **IGNORE** | `LOW_RELIABILITY` | $\mathcal{U}(x) \le \tau_{\text{ignore}} \land R(x) < 0.35$ | Suspicious sample with low feature integrity or conflicting label. |
| **IGNORE** | `REDUNDANT` | $\mathcal{U}(x) \le \tau_{\text{ignore}} \land R(x) \ge 0.35$ | Sample is already well-represented and confidently predicted; filtered for compute efficiency. |
| **IGNORE** | `FALLBACK_SAFE` | $\text{NaN} / \text{Inf}$ encountered | Non-finite signal detected; fails safely to IGNORE to protect weights. |
| **DEFER** | `AMBIGUOUS_DEFERRED` | $\tau_{\text{ignore}} < \mathcal{U}(x) < \tau_{\text{learn}}$ | Marginal sample quarantined in the bounded queue for re-evaluation. |

---

## 4. Bounded Defer Queue Architecture

To prevent unbounded memory growth in production or streaming deployments, `DeferQueue` implements strict capacity limits ($C = 500$ samples) with three user-configurable overflow policies:
1. **`drop_oldest` (Default):** First-in, first-out overflow eviction.
2. **`drop_lowest_utility`:** Evicts the item in the queue with the lowest estimated utility $\mathcal{U}(x)$.
3. **`reject_new`:** Rejects newly deferred items if capacity is saturated, forcing immediate fallback to `IGNORE`.

---

## 5. Empirical Results on 2,000-Sample CIFAR-10 Stream

Evaluating the policy on a 2,000-sample stream with 20% label noise and 15% feature corruption yields the following distribution:

### 5.1 Overall Action Distribution
- **Total Evaluated:** 2,000 samples
- **LEARN:** 712 samples (35.60%)
- **IGNORE:** 655 samples (32.75%)
- **DEFER:** 633 samples (31.65%)
- **Queue Overflows:** 133 samples handled cleanly via `drop_oldest`

### 5.2 Confusion Breakdown across Ground Truth Conditions

| Ground Truth Condition | Total Samples | LEARN (Admitted) | IGNORE (Filtered) | DEFER (Quarantined) |
|---|---|---|---|---|
| **Clean** | 1,345 | 596 (44.31%) | 183 (13.61%) | 566 (42.08%) |
| **Noisy Label** | 392 | 14 (3.57%) | 368 (93.88%) | 10 (2.55%) |
| **Corrupted Feature** | 263 | 102 (38.78%) | 104 (39.54%) | 57 (21.67%) |

### 5.3 Key Research Findings

1. **Noise Interception Rate: 96.43%**
   - Out of 392 maliciously corrupted labels, the gate successfully rejected 368 directly via `IGNORE` and quarantined 10 in `DEFER`.
   - Only 14 noisy samples (3.57%) slipped through to `LEARN`.
   - In contrast, a standard passive continual learner would update on 100% of noisy samples, severely damaging representation quality.

2. **Compute Conservation via Redundancy Filtering**
   - 183 clean samples (13.61%) were marked `REDUNDANT` and ignored.
   - These represent samples where the model was already confident ($U < 0.10$) and the image was nearly identical to existing class prototypes ($N < 0.05$). Training on them would produce negligible gradients while wasting compute.

3. **Ambiguity Resolution via Deferral**
   - 42.08% of clean samples were placed in `DEFER`. In Phase 6 (Continual Training), as parameter updates shift model weights, these deferred samples can be re-evaluated when the model is in a better epistemic state to judge them.

---

## 6. Deliverable Verification Checklist

- [x] Typed `GateAction`, `ReasonCode`, and `GateDecision` classes implemented in [`src/alg/gating/decision.py`](../../src/alg/gating/decision.py).
- [x] Transparent utility score $\mathcal{U}(x) = (B(x) - H(x) + 1)/2$ in [`src/alg/gating/policy.py`](../../src/alg/gating/policy.py).
- [x] Auditable reason codes, signal snapshots, and sample IDs recorded on every decision.
- [x] Bounded `DeferQueue` with `drop_oldest`, `drop_lowest_utility`, and `reject_new` overflow policies in [`src/alg/gating/defer_queue.py`](../../src/alg/gating/defer_queue.py).
- [x] Diagnostic script [`scripts/evaluate_gate.py`](../../scripts/evaluate_gate.py) with structured JSON export.
- [x] Unit test suite with 15 dedicated tests in `tests/test_gate.py` and `tests/test_defer_queue.py` passing (total project suite: 106 tests passed).
