# V5 experiment completion addendum

Status: the remaining preregistered `Geometry-global` CASP15 cell is complete.
This is a development analysis and does not modify the Raw pipeline frozen for
the Kaggle scorer check.

## Geometry-global configuration

The configuration keeps the historical Geometry objective and adaptive source
strength, changes angle/torsion context from candidate-derived to the
training-derived global prior, and disables the radius-of-gyration term. It was
run on the same four frozen candidate banks and the same five candidates per
bank as the existing refinement factorial. All 12 targets completed with zero
generation and evaluation failures.

## Main V3 3T+2D results

| Comparison | Metric | Geometry-global | Comparator | Favourable delta | 95% cluster-bootstrap CI | Wins/ties/losses |
|---|---|---:|---:|---:|---:|---:|
| Geometry-global vs Simple | SW-RMSD9 | 4.103093 A | 4.139421 A | 0.036328 A reduction | [0.025812, 0.047563] | 12/0/0 |
| Geometry-global vs Simple | SW-RMSD15 | 7.009695 A | 7.027153 A | 0.017458 A reduction | [0.011357, 0.023678] | 12/0/0 |
| Geometry-global vs Simple | C1-prime lDDT | 0.465742 | 0.469207 | -0.003465 | [-0.005072, -0.002019] | 1/0/11 |
| Geometry-global vs Simple | locked-reference candidate TM | 0.313756 | 0.314448 | -0.000692 | [-0.002764, 0.000969] | 4/0/8 |
| Geometry-global vs Simple | bank best-of-five TM | 0.463786 | 0.463248 | +0.000537 | [-0.001139, 0.002602] | 9/0/3 |
| Geometry-global vs Raw | bank best-of-five TM | 0.463786 | 0.464384 | -0.000598 | [-0.003405, 0.001116] | 9/0/3 |
| Geometry-global vs historical Geometry | bank best-of-five TM | 0.463786 | 0.464267 | -0.000481 | [-0.003809, 0.002418] | 5/0/7 |

The result reinforces the existing RQ3 interpretation. Geometry-global repairs
the SW-RMSD windows beyond Simple, but lDDT moves in the opposite direction and
there is no demonstrated best-of-five TM improvement. It is essentially tied
with historical Geometry on SW-RMSD9 and has a slightly lower point-estimate TM.
Consequently it does not change selection: the TM-optimized frozen deployment
remains exact reconstructed V3 `3T+2D Raw`.

## Artifact boundary

- Generation and evaluation outputs are under `geometry_global_secondary/`.
- The original refinement factorial and `V5_VERIFIED_EVIDENCE.json` remain
  byte-frozen as the evidence used for pre-Kaggle method selection.
- This addendum closes an experiment-matrix omission without retroactively
  changing the scientific method or its selection evidence.
