# Benchmark Empirical Results Summary

| Condition | Method | Test Acc (%) | Noise Ingested | Noise Blocked (%) | Compute Saved (%) | Runtime (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `high_noise_stream` | Core-Set Diversity (Novelty-Only) | 90.35 ± 0.00 | 2040/3974 | 48.7% | 48.1% | 38.2s |
| `high_noise_stream` | Adaptive Learning Gate (ALG Ours) | 91.46 ± 0.00 | 21/3974 | 99.5% | 97.4% | 37.7s |
| `high_noise_stream` | Budget-Matched Random | 91.09 ± 0.00 | 1517/3974 | 61.8% | 62.0% | 38.2s |
| `high_noise_stream` | Small-Loss (Co-Teaching / MentorNet) | 92.18 ± 0.00 | 30/3974 | 99.2% | 43.1% | 38.6s |
| `high_noise_stream` | Active Learning (Uncertainty-Only) | 92.38 ± 0.00 | 419/3974 | 89.5% | 89.6% | 37.6s |
| `high_noise_stream` | Conventional (Update-All) | 91.94 ± 0.00 | 3974/3974 | 0.0% | 0.0% | 37.4s |
| `noisy_stream` | ALG w/o Deferral (Hard Binary Gate) | 93.89 ± 0.00 | 13/1925 | 99.3% | 22.5% | 37.4s |
| `noisy_stream` | ALG w/o Novelty (w_nov=0) | 93.11 ± 0.00 | 9/1925 | 99.5% | 97.9% | 39.0s |
| `noisy_stream` | ALG w/o Reliability (w_unrel=0) | 93.07 ± 0.00 | 1103/1925 | 42.7% | 42.5% | 41.4s |
| `noisy_stream` | ALG w/o Uncertainty (w_unc=0) | 93.34 ± 0.00 | 7/1925 | 99.6% | 98.6% | 35.8s |
| `noisy_stream` | Core-Set Diversity (Novelty-Only) | 93.52 ± 0.00 | 541/1925 | 71.9% | 71.5% | 37.6s |
| `noisy_stream` | Adaptive Learning Gate (ALG Ours) | 93.75 ± 0.00 | 7/1925 | 99.6% | 97.5% | 40.7s |
| `noisy_stream` | Budget-Matched Random | 92.70 ± 0.00 | 706/1925 | 63.3% | 62.0% | 38.0s |
| `noisy_stream` | Small-Loss (Co-Teaching / MentorNet) | 93.03 ± 0.00 | 10/1925 | 99.5% | 22.5% | 39.6s |
| `noisy_stream` | Active Learning (Uncertainty-Only) | 92.95 ± 0.00 | 82/1925 | 95.7% | 95.8% | 37.6s |
| `noisy_stream` | Conventional (Update-All) | 93.03 ± 0.00 | 1925/1925 | 0.0% | 0.0% | 39.0s |