# TBM white-box component ablation (20 calibration RNA)

This mechanism study uses only the 20 calibration RNAs. Every template must be released strictly before its target and the target PDB itself is excluded. The `unsafe_dates` row is shown only to quantify temporal leakage and is never a method.

## Search source

TM values for MMseqs are conditional on finding at least one template; availability is reported separately so 12 missing targets cannot silently disappear from the mean.

| source_variant        |   n_targets |   available_targets |   availability |   mean_candidates_all_targets |   top1_tm_when_available |   best5_tm_when_available |   mean_coverage_when_available |
|:----------------------|------------:|--------------------:|---------------:|------------------------------:|-------------------------:|--------------------------:|-------------------------------:|
| MMseqs_only           |          20 |                   8 |            0.4 |                             2 |                 0.660029 |                  0.678316 |                       0.930965 |
| MMseqs_plus_composite |          20 |                  20 |            1   |                             5 |                 0.376955 |                  0.389705 |                       0.812411 |
| composite_only        |          20 |                  20 |            1   |                             5 |                 0.334934 |                  0.37606  |                       0.754893 |

## Composite retrieval-score components

`useful_hit_045` is the fraction of targets whose retrieved top five contain a raw template candidate with TM ≥ 0.45. `useful_recall_045` is recall inside the union of candidates scored by this component study; it is a diagnostic, not an official biological metric.

| variant                    |   top1_tm |   best5_tm |   mean_coverage |   useful_hit_045 |   useful_recall_045 |
|:---------------------------|----------:|-----------:|----------------:|-----------------:|--------------------:|
| G_L_F                      |  0.335791 |   0.37923  |        0.734794 |             0.35 |            0.676304 |
| G_L_F_K_equal              |  0.34137  |   0.37656  |        0.778492 |             0.35 |            0.635488 |
| G_only                     |  0.341604 |   0.381177 |        0.808214 |             0.35 |            0.635488 |
| G_plus_L                   |  0.336936 |   0.380465 |        0.726099 |             0.35 |            0.676304 |
| full_weighted              |  0.334934 |   0.37606  |        0.754893 |             0.35 |            0.676304 |
| full_weighted_unsafe_dates |  0.33568  |   0.403022 |        0.774664 |             0.35 |            0.696145 |
| wG_minus25pct              |  0.335212 |   0.375436 |        0.738686 |             0.35 |            0.676304 |
| wG_plus25pct               |  0.3411   |   0.3765   |        0.757319 |             0.35 |            0.676304 |
| wL_minus25pct              |  0.341073 |   0.3765   |        0.761148 |             0.35 |            0.635488 |
| wL_plus25pct               |  0.336994 |   0.376312 |        0.730597 |             0.35 |            0.676304 |

## Reranking and distinct-PDB selection

| ranking                            |   top1_tm |   top3_tm |   top5_tm |   mean_coverage |   n_distinct_pdb |
|:-----------------------------------|----------:|----------:|----------:|----------------:|-----------------:|
| full_plus_distinct_pdb             |   0.36372 |  0.378412 |  0.387558 |        0.935347 |             5    |
| identity_only                      |   0.33445 |  0.363854 |  0.3719   |        0.661799 |             4.85 |
| identity_x_coverage                |   0.36372 |  0.378412 |  0.387558 |        0.933495 |             4.95 |
| identity_x_coverage_x_completeness |   0.36372 |  0.378412 |  0.387558 |        0.933495 |             4.95 |

## Paired target effects

Every delta is oriented so positive means the named method is better. Intervals bootstrap RNA targets (10,000 samples), not templates or residues.

| effect                               | metric   | baseline                           | method                             |   n_targets |   mean_delta |   median_delta |    ci_low |   ci_high |   improved |   tied |   regressed |
|:-------------------------------------|:---------|:-----------------------------------|:-----------------------------------|------------:|-------------:|---------------:|----------:|----------:|-----------:|-------:|------------:|
| G-only versus full weighted          | best5_tm | full_weighted                      | G_only                             |          20 |     0.005117 |              0 | -0.002178 |  0.015004 |          5 |     12 |           3 |
| temporal leakage (unsafe minus safe) | best5_tm | full_weighted                      | full_weighted_unsafe_dates         |          20 |     0.026962 |              0 | -0.000343 |  0.071258 |          3 |     15 |           2 |
| add target coverage to identity      | top5_tm  | identity_only                      | identity_x_coverage                |          20 |     0.015659 |              0 | -0.003731 |  0.045178 |          7 |      6 |           7 |
| add template completeness            | top5_tm  | identity_x_coverage                | identity_x_coverage_x_completeness |          20 |     0        |              0 |  0        |  0        |          0 |     20 |           0 |
| enforce distinct PDB                 | top5_tm  | identity_x_coverage_x_completeness | full_plus_distinct_pdb             |          20 |     0        |              0 |  0        |  0        |          0 |     20 |           0 |
| add MMseqs candidates to composite   | best5_tm | composite_only                     | MMseqs_plus_composite              |          20 |     0.013645 |              0 | -0.004417 |  0.042781 |          2 |     16 |           2 |

## Gap filling on unsupported residues

Values are 9-residue sliding-window C1′ RMSD at unsupported residues. Positive paired deltas mean the current curved-gap heuristic is better than linear filling. Candidate gaps are averaged within each RNA before the target bootstrap.

| gap_bin   |   linear_sw9 |   current_sw9 |   n_targets |   mean_delta |   median_delta |    ci_low |   ci_high |   improved |   tied |   regressed |
|:----------|-------------:|--------------:|------------:|-------------:|---------------:|----------:|----------:|-----------:|-------:|------------:|
| 1-3       |      4.54607 |       4.54712 |          19 |    -0.001051 |              0 | -0.008922 |  0.006189 |          2 |     15 |           2 |
| 4-8       |      5.65257 |       5.6692  |          12 |    -0.016628 |              0 | -0.106042 |  0.044837 |          5 |      4 |           3 |
| 9-20      |      8.05516 |       8.01838 |          15 |     0.036778 |              0 |  0.009289 |  0.067605 |          5 |      9 |           1 |
| >20       |      8.12242 |       8.12242 |           6 |     0        |              0 |  0        |  0        |          0 |      6 |           0 |
