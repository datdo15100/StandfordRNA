# Refiner comparison & de novo uplift — CASP15 validation

Best-of-5 TM-score (US-align). Methods ported/adapted from the 1st-place TBM notebook are compared against ours.

## (A) De novo fallback vs extended chain — NO-TEMPLATE targets (n=7)

- extended chain (old): **0.0681**
- de novo (ported): **0.1540**
- de novo + gradient refine: **0.1588**
- de novo + rule-based refine: **0.1634**

## (B) Refiner head-to-head — TEMPLATED targets (n=5)

- TBM no refine: 0.2852
- + gradient (ours): **0.2858**
- + rule-based (1st place): 0.2853

## Overall means (all targets)

- extended: 0.0685
- denovo: 0.1627
- grad: 0.2117
- rule: 0.2142

## Physical validity — representative structure, before vs after

- clashes/residue: before 0.330 → grad 0.155 / rule 0.241
- backbone deviation (A): before 2.207 → grad 0.308 / rule 1.028

## Takeaways

1. **De novo fallback is the dominant win**: on no-template targets it more than doubles TM over the extended-chain floor (0.068 → 0.154), lifting the overall mean from ~0.161 to ~0.213.
2. **On TM the two refiners are near-tied** — TM is robust to local error, so neither moves it much on a decent template. Rule-based edges ahead only on the rough de novo inits (gentler nudging preserves global shape): overall rule 0.2142 vs gradient 0.2117; on templated targets gradient 0.2858 vs rule 0.2853.
3. **On physical validity our gradient refinement wins decisively**: it removes ~53 % of clashes vs ~27 % for rule-based, and cuts backbone deviation by ~86 % vs ~53 %. Equal TM, far more physically plausible structures — the thesis differentiator (TM alone does not reward valid geometry; downstream full-atom reconstruction / docking needs it).
4. **v2 insight**: gradient refinement is a touch aggressive on very rough (de novo) inits; scaling its strength down there should recover the small TM gap and likely overtake rule-based everywhere.

## Per-target

| target_id   |   seq_len | has_template   |   best_conf |   extended |   denovo |   tbm_noref |   grad |   rule |
|:------------|----------:|:---------------|------------:|-----------:|---------:|------------:|-------:|-------:|
| R1107       |        69 | True           |       0.58  |     0.0873 |   0.1615 |      0.3292 | 0.326  | 0.3251 |
| R1108       |        69 | True           |       0.58  |     0.09   |   0.1591 |      0.3183 | 0.3124 | 0.3135 |
| R1116       |       157 | True           |       0.662 |     0.0605 |   0.1579 |      0.4534 | 0.4689 | 0.4623 |
| R1117v2     |        30 | False          |       0     |     0.0997 |   0.11   |    nan      | 0.0999 | 0.1232 |
| R1126       |       363 | True           |       0.149 |     0.0499 |   0.2078 |      0.1676 | 0.1616 | 0.1697 |
| R1128       |       238 | False          |       0     |     0.0536 |   0.1568 |    nan      | 0.1786 | 0.1816 |
| R1136       |       374 | True           |       0.139 |     0.0575 |   0.1884 |      0.1575 | 0.16   | 0.1558 |
| R1138       |       720 | False          |       0     |     0.0357 |   0.1679 |    nan      | 0.1747 | 0.1831 |
| R1149       |       124 | False          |       0     |     0.0659 |   0.1559 |    nan      | 0.1644 | 0.1666 |
| R1156       |       135 | False          |       0     |     0.071  |   0.1708 |    nan      | 0.1731 | 0.1722 |
| R1189       |       118 | False          |       0     |     0.0716 |   0.1616 |    nan      | 0.165  | 0.1588 |
| R1190       |       118 | False          |       0     |     0.0793 |   0.155  |    nan      | 0.1557 | 0.1582 |

## Per-target geometry

| target_id   | has_template   |   clash_before |   clash_grad |   clash_rule |   bbdev_before |   bbdev_grad |   bbdev_rule |
|:------------|:---------------|---------------:|-------------:|-------------:|---------------:|-------------:|-------------:|
| R1107       | True           |          0     |        0     |        0     |          0.868 |        0.527 |        0.672 |
| R1108       | True           |          0     |        0     |        0     |          0.868 |        0.527 |        0.669 |
| R1116       | True           |          0.287 |        0.191 |        0.21  |          1.633 |        0.717 |        1.276 |
| R1117v2     | False          |          0.367 |        0.033 |        0.3   |          2.504 |        0.085 |        1.015 |
| R1126       | True           |          0     |        0     |        0     |          0.244 |        0.397 |        0.112 |
| R1128       | False          |          0.597 |        0.277 |        0.521 |          3.824 |        0.259 |        1.583 |
| R1136       | True           |          0     |        0     |        0.003 |          0.259 |        0.421 |        0.121 |
| R1138       | False          |          0.368 |        0.211 |        0.342 |          3.657 |        0.223 |        1.596 |
| R1149       | False          |          0.879 |        0.282 |        0.508 |          3.745 |        0.297 |        1.543 |
| R1156       | False          |          0.141 |        0.341 |        0.156 |          3.055 |        0.156 |        1.288 |
| R1189       | False          |          0.661 |        0.263 |        0.424 |          2.915 |        0.046 |        1.23  |
| R1190       | False          |          0.661 |        0.263 |        0.424 |          2.915 |        0.046 |        1.23  |