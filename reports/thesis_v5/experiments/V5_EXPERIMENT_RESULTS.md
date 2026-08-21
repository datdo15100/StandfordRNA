# V5 CASP15 experiment results

Status: complete local development/validation evidence. This is not a thesis chapter.

## Main results

| Question | Comparison | Mean scores | Paired delta | 95% cluster-bootstrap CI | Wins/ties/losses |
|---|---|---:|---:|---:|---:|
| RQ1 | exact V3 TBM Raw vs J-SS-RAW | 0.309164 vs 0.297900 | 0.011264 (3.78%) | [0.006948, 0.015806] | 12/0/0 |
| RQ2 | V3 3T+2D Raw vs V3 5T Raw | 0.464384 vs 0.309164 | 0.155220 (50.21%) | [0.056134, 0.304327] | 10/2/0 |
| RQ3 local | Geometry vs Simple, SW-RMSD9 | 4.103914 vs 4.139421 A | 0.035507 A reduction | [0.025265, 0.046653] | 12/0/0 |
| RQ3 global | Geometry vs Simple, best-of-five TM | 0.464267 vs 0.463248 | 0.001018 | [-0.001246, 0.004252] | 9/0/3 |

## Interpretation locked to the evidence

- RQ1: the reconstructed thesis TBM improves local CASP15 best-of-five TM over the inherited John algorithm in the common sandbox.
- The clearest internal TBM Lego is identity x query coverage: +0.013347 TM. MMseqs retrieval, completeness, and distinct-PDB selection produce no TM change on this 12-target cohort; they are safeguards/availability mechanisms here.
- RQ2: direct DRfold2 candidates are much stronger than template-only candidates on CASP15. The 2D bank already reaches 0.449239; adding three templates raises this to 0.464384, a further 0.015145.
- The structural-diversity mechanism is weak: mean self-TM changes from 0.266313 for 5T to 0.264470 for 3T+2D. The large TM gain is therefore better described as pretrained-source accuracy/coverage plus a smaller template complement, not as proven generic diversity.
- RQ3: Geometry improves SW-RMSD9 beyond Simple by 0.035507 A on all 12 targets, but C1'-lDDT changes by -0.003714 in the opposite direction. Its bank TM delta versus Simple is small and uncertain. This is a metric-specific local trade-off, not universal refinement superiority.
- The selected TM-optimized deployment is `V3 exact reconstructed 3T+2D Raw` at 0.464384 CASP15 TM. Geometry remains an analyzed local-refinement contribution but is not in the selected deployment.

## Kaggle limitation discovered before submission

`validation_sequences.csv` and the downloaded `test_sequences.csv` are byte-identical (42dc2e35aa92c2bd3d3d32a43bd505a3f12568d2ecde080d9783fb3e2eaa89e0). Therefore a late Kaggle score obtained after these CASP15 experiments cannot honestly be called an independent hidden-target test. It may still be reported as an official-scorer/deployment compatibility check.
