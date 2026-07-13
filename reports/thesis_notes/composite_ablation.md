# Composite-search fallback ablation — CASP15 (temporal-safe, best-of-5 TM)

`comp_on` merges MMseqs hits with an exhaustive composite-similarity scan and
re-ranks the combined pool by confidence. Unused best-of-five slots retain a
de novo hedge. Both branches use the same temporal and self-leakage filters.

- **comp_off (MMseqs only, previous): 0.2117**
- **comp_on (MMseqs + composite fallback): 0.3072**  (Δ **+0.0955**)
- top-1 reproduced (temporal-safe): 0.2983

Targets improved: 11, unchanged 0, worse 1 (R1116, −0.0084). The current method
beats the freshly reproduced top-1 baseline on 9/12 targets. The gain comes from
better template recall; the gradient refiner is unchanged.

| target_id   |   seq_len |   comp_off |   comp_on |   delta |   n_cand_off |   n_cand_on |   n_composite |   top1_tsafe |   sec |
|:------------|----------:|-----------:|----------:|--------:|-------------:|------------:|--------------:|-------------:|------:|
| R1107       |        69 |     0.326  |    0.3805 |  0.0545 |            5 |           5 |             1 |       0.3722 |   6.4 |
| R1108       |        69 |     0.3124 |    0.4889 |  0.1765 |            5 |           5 |             1 |       0.4767 |   5.1 |
| R1116       |       157 |     0.4689 |    0.4605 | -0.0084 |            5 |           5 |             3 |       0.5281 |   6.1 |
| R1117v2     |        30 |     0.0999 |    0.4199 |  0.32   |            0 |           5 |             5 |       0.414  |   4.3 |
| R1126       |       363 |     0.1616 |    0.2062 |  0.0446 |            5 |           5 |             5 |       0.2058 |   8.5 |
| R1128       |       238 |     0.1786 |    0.2251 |  0.0465 |            0 |           5 |             5 |       0.1569 |   7   |
| R1136       |       374 |     0.16   |    0.2321 |  0.0721 |            5 |           5 |             5 |       0.1878 |   9.2 |
| R1138       |       720 |     0.1747 |    0.2257 |  0.051  |            0 |           5 |             5 |       0.2195 |  33.8 |
| R1149       |       124 |     0.1644 |    0.3243 |  0.1599 |            0 |           5 |             5 |       0.2971 |   7.4 |
| R1156       |       135 |     0.1731 |    0.2795 |  0.1064 |            0 |           5 |             5 |       0.2716 |  11.3 |
| R1189       |       118 |     0.165  |    0.2219 |  0.0569 |            0 |           5 |             5 |       0.2233 |   6.6 |
| R1190       |       118 |     0.1557 |    0.2221 |  0.0664 |            0 |           5 |             5 |       0.2269 |   6.6 |
