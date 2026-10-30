# Benchmark Empirical Results Summary

| Condition | Method | Test Acc (%) | Noise Ingested | Noise Blocked (%) | Compute Saved (%) | Runtime (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `noisy_stream` | ALG w/o Deferral (Hard Binary Gate) | 85.11 ± 0.00 | 0/13 | 100.0% | 20.0% | 12.5s |
| `noisy_stream` | ALG w/o Novelty (w_nov=0) | 94.34 ± 0.00 | 0/13 | 100.0% | 100.0% | 12.5s |
| `noisy_stream` | ALG w/o Reliability (w_unrel=0) | 94.34 ± 0.00 | 0/13 | 100.0% | 99.0% | 12.6s |
| `noisy_stream` | ALG w/o Uncertainty (w_unc=0) | 94.34 ± 0.00 | 0/13 | 100.0% | 100.0% | 12.3s |
| `noisy_stream` | Core-Set Diversity (Novelty-Only) | 94.34 ± 0.00 | 0/13 | 100.0% | 100.0% | 12.7s |
| `noisy_stream` | Adaptive Learning Gate (ALG Ours) | 94.34 ± 0.00 | 0/13 | 100.0% | 100.0% | 13.2s |
| `noisy_stream` | Budget-Matched Random | 77.06 ± 0.00 | 5/13 | 61.5% | 56.0% | 13.2s |
| `noisy_stream` | Small-Loss (Co-Teaching / MentorNet) | 85.11 ± 0.00 | 0/13 | 100.0% | 20.0% | 12.6s |
| `noisy_stream` | Active Learning (Uncertainty-Only) | 94.34 ± 0.00 | 0/13 | 100.0% | 100.0% | 12.5s |
| `noisy_stream` | Conventional (Update-All) | 70.83 ± 0.00 | 13/13 | 0.0% | 0.0% | 13.2s |