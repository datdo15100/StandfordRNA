# TBM–DRfold2 complementarity on 20 newest held-out RNA

This report reads the already frozen validation artifacts; it does not rerun a model or access native labels again. Candidate selection is confidence-based and native blind. Oracle rows use native scores only to measure the available ceiling.

- Mean best individual TBM candidate: **0.565183**
- Mean best individual DRfold2 candidate: **0.502627**
- Mean union oracle: **0.588304**
- DRfold2/TBM/tie oracle winner counts: **8/12/0**

## Candidate allocation (Geometry off)

|   total_candidates |   n_tbm |   n_drfold2 |     mean |      std |   count |
|-------------------:|--------:|------------:|---------:|---------:|--------:|
|                  1 |       0 |           1 | 0.488402 | 0.174709 |      20 |
|                  1 |       1 |           0 | 0.533634 | 0.259215 |      20 |
|                  2 |       0 |           2 | 0.502627 | 0.176008 |      20 |
|                  2 |       1 |           1 | 0.579586 | 0.213909 |      20 |
|                  2 |       2 |           0 | 0.54904  | 0.244857 |      20 |
|                  3 |       1 |           2 | 0.582752 | 0.210272 |      20 |
|                  3 |       2 |           1 | 0.581196 | 0.216359 |      20 |
|                  3 |       3 |           0 | 0.565183 | 0.233256 |      20 |
|                  5 |       3 |           2 | 0.588304 | 0.213796 |      20 |

## Paired target effects

Positive means the method named after the colon is better. The fixed-N rows separate source composition from candidate-count advantage; the production-bank row is pragmatic but receives two additional prediction slots.

| effect                                        |   n_targets |   mean_delta |   median_delta |    ci_low |   ci_high |   improved |   tied |   regressed |
|:----------------------------------------------|------------:|-------------:|---------------:|----------:|----------:|-----------:|-------:|------------:|
| fixed-N=3: replace one TBM with one DRfold2   |          20 |     0.016012 |       0        | -0.002072 |  0.041466 |          7 |     10 |           3 |
| production bank: add two DRfold2 to three TBM |          20 |     0.023121 |       0        |  0.006444 |  0.047519 |          8 |     12 |           0 |
| fixed-N=2: replace one TBM with one DRfold2   |          20 |     0.030546 |       0        |  0.002949 |  0.068185 |          7 |     11 |           2 |
| two DRfold2 versus two TBM                    |          20 |    -0.046414 |      -0.031225 | -0.105771 |  0.017253 |          8 |      0 |          12 |
| oracle ceiling: union versus TBM              |          20 |     0.023121 |       0        |  0.006444 |  0.047519 |          8 |     12 |           0 |

## Interpretation

DRfold2 is not uniformly stronger than TBM. Its value is complementary coverage: on a minority of targets it supplies a fold absent from the TBM bank. A fixed-size hybrid allocation has a positive mean effect but a wide interval, while the union oracle quantifies how much a perfect native-blind selector could in principle gain.
