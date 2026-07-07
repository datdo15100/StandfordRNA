# Leakage demonstration — CASP15 validation (TBM + refinement)

Best-of-5 TM under three template-availability regimes.

- **temporal_safe (honest): 0.1612**
- no_temporal (ignore cutoff, exclude native pdb): 0.6388
- oracle_leak (native allowed): 0.9566

Temporal discipline costs **+0.4776** TM vs ignoring the cutoff, and **+0.7954** vs full leakage. The oracle column confirms the TBM+refinement machinery reaches high TM when a true template is available — the honest score is bounded by template availability, not the pipeline.

| target_id   |   seq_len |   temporal_safe |   no_temporal |   oracle_leak |
|:------------|----------:|----------------:|--------------:|--------------:|
| R1107       |        69 |          0.326  |        0.7235 |        0.9952 |
| R1108       |        69 |          0.3124 |        0.7209 |        0.9951 |
| R1116       |       157 |          0.4689 |        0.7213 |        0.9962 |
| R1117v2     |        30 |          0.1074 |        0.1074 |        0.9369 |
| R1126       |       363 |          0.1616 |        0.2434 |        0.9892 |
| R1128       |       238 |          0.0488 |        0.0488 |        0.9996 |
| R1136       |       374 |          0.16   |        0.9693 |        0.9996 |
| R1138       |       720 |          0.0377 |        0.9999 |        0.9999 |
| R1149       |       124 |          0.0697 |        0.8388 |        0.9331 |
| R1156       |       135 |          0.0767 |        0.8525 |        0.8908 |
| R1189       |       118 |          0.0789 |        0.7081 |        0.8705 |
| R1190       |       118 |          0.0867 |        0.7318 |        0.8731 |