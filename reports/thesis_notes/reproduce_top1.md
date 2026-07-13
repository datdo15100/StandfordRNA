# 1st-place TBM-only — faithful reproduction on the 12 CASP15 targets

Best-of-5 TM (US-align). Same method (composite similarity + KMeans diversity + transfer + rule-based refine + de novo), scored under two template regimes.

- **full_pdb (their setup — no temporal filter, LEAKED): 0.9355**
- **temporal_safe (honest): 0.2983**

Leakage on CASP15 = **+0.6372** TM. Their public 0.593 is a *private-set* score (≈40 hidden targets), NOT reproducible on these 12 public targets; full_pdb here is the local leaked proxy.

## What this reproduction diagnosed

Before composite search, our MMseqs + de novo pipeline scored 0.2117. The reproduced
top-1 method scored 0.2983 temporal-safe because its exhaustive composite similarity
scan returns plausible real-fold templates even when MMseqs finds no homolog. This
isolated candidate recall—not coordinate refinement—as the primary bottleneck.

After adding the same class of composite search under our temporal/self-leakage
controls, the current pipeline reaches 0.3072 and beats this reproduction on 9/12
targets. Thus this report is both a strong baseline and the diagnostic that motivated
the current search improvement.

## Caveats

- The reconstructed library contains 7,155 unique sequences from our gemmi-parsed
  database, not the top-1 notebook's exact 18,881-entry extraction.
- `full_pdb` deliberately includes post-cutoff/native structures and is not an honest
  local validation result.
- Only a Kaggle late submission can produce a score on the hidden private targets.

| target_id   |   seq_len |   full_pdb |   full_pdb_sec |   temporal_safe |   temporal_safe_sec |
|:------------|----------:|-----------:|---------------:|----------------:|--------------------:|
| R1107       |        69 |     0.9944 |            4.4 |          0.3722 |                 2.9 |
| R1108       |        69 |     0.9943 |            4.4 |          0.4767 |                 2.9 |
| R1116       |       157 |     0.9949 |            5.3 |          0.5281 |                 3.4 |
| R1117v2     |        30 |     0.9539 |            3.2 |          0.414  |                 2.4 |
| R1126       |       363 |     0.9886 |            8.7 |          0.2058 |                 5   |
| R1128       |       238 |     0.9994 |            6.1 |          0.1569 |                 3.9 |
| R1136       |       374 |     0.9689 |            9.2 |          0.1878 |                 5.7 |
| R1138       |       720 |     0.9999 |           34.1 |          0.2195 |                24.4 |
| R1149       |       124 |     0.9343 |            6.9 |          0.2971 |                 4.8 |
| R1156       |       135 |     0.8178 |           11.4 |          0.2716 |                 8.7 |
| R1189       |       118 |     0.7066 |            6.2 |          0.2233 |                 4.1 |
| R1190       |       118 |     0.8734 |            6.1 |          0.2269 |                 4.1 |
