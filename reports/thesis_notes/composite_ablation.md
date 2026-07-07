# Composite-search ablation — CASP15 (temporal-safe, best-of-5 TM)

`comp_on` = MMseqs **+ exhaustive composite similarity, always merged** and re-ranked by
confidence (keeps the best template from either source), with a de novo hedge.

- **comp_off (MMseqs only, previous): 0.2117**
- **comp_on (MMseqs + composite search): 0.3072**  (Δ **+0.0955**)
- top-1 reproduced (temporal-safe): 0.2973 → **we now edge past it (+0.010)**

Targets improved: 11, unchanged 0, worse 1 (R1116, −0.008). We beat the reproduced
1st-place method on 9 of 12 targets; it stays ahead only on R1116 and marginally on
R1189/R1190. The entire gain comes from better template *recall* — the gradient
refinement is unchanged.

| target_id   |   seq_len |   comp_off |   comp_on |   delta |   n_cand_off |   n_cand_on |   n_composite |   top1_tsafe |   sec |
|:------------|----------:|-----------:|----------:|--------:|-------------:|------------:|--------------:|-------------:|------:|
| R1107       |        69 |     0.326  |    0.3805 |  0.0545 |            5 |           5 |             1 |       0.3709 |  10.8 |
| R1108       |        69 |     0.3124 |    0.4889 |  0.1765 |            5 |           5 |             1 |       0.4749 |   8.8 |
| R1116       |       157 |     0.4689 |    0.4605 | -0.0084 |            5 |           5 |             3 |       0.5262 |   9.8 |
| R1117v2     |        30 |     0.0999 |    0.4199 |  0.32   |            0 |           5 |             5 |       0.3946 |   6.8 |
| R1126       |       363 |     0.1616 |    0.2062 |  0.0446 |            5 |           5 |             5 |       0.18   |  14.6 |
| R1128       |       238 |     0.1786 |    0.2251 |  0.0465 |            0 |           5 |             5 |       0.1995 |  11.6 |
| R1136       |       374 |     0.16   |    0.2321 |  0.0721 |            5 |           5 |             5 |       0.1902 |  15.2 |
| R1138       |       720 |     0.1747 |    0.2257 |  0.051  |            0 |           5 |             5 |       0.2173 |  50.9 |
| R1149       |       124 |     0.1644 |    0.3243 |  0.1599 |            0 |           5 |             5 |       0.2957 |  11.7 |
| R1156       |       135 |     0.1731 |    0.2795 |  0.1064 |            0 |           5 |             5 |       0.2622 |  16   |
| R1189       |       118 |     0.165  |    0.2219 |  0.0569 |            0 |           5 |             5 |       0.2255 |  10.3 |
| R1190       |       118 |     0.1557 |    0.2221 |  0.0664 |            0 |           5 |             5 |       0.2304 |  10.9 |