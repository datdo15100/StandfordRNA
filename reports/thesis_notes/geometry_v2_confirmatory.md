# Geometry v2 confirmatory validation (20 newest RNA)

- Frozen method: **source_1p5**
- Confirmatory independent-metric gate: **pass**
- Split protocol: 60 train-prior → 20 calibration → 20 newest validation, with no sequence/family group crossing a split.
- Native reference for each refined candidate is fixed from its raw candidate, so refinement cannot benefit by switching to an easier native conformation.
- All headline means and confidence intervals give equal weight to each RNA target.

## Equal-weight target means

| setting                |   best5_tm |   c1_rmsd |   c1_lddt |   sw_rmsd_9 |   sw_rmsd_15 |   sw_rmsd_31 |   bb_dev |   clash_per_res |   rg_err |   sharp_kinks |   angle_nll |   torsion_nll |   pair_like_fraction |   mean_drift |
|:-----------------------|-----------:|----------:|----------:|------------:|-------------:|-------------:|---------:|----------------:|---------:|--------------:|------------:|--------------:|---------------------:|-------------:|
| raw                    |   0.588304 |   10.9036 |  0.659836 |     2.65139 |      4.27339 |      7.56265 | 1.04027  |        0.065121 |  6.17411 |      0.048977 |    0.820552 |      0.837811 |             0.612257 |     0        |
| simple_source_backbone |   0.587767 |   10.903  |  0.667417 |     2.64159 |      4.26478 |      7.55855 | 0.837615 |        0.040708 |  6.17076 |      0.068558 |    0.869599 |      0.871896 |             0.607337 |     0.164081 |
| source_1p5             |   0.587197 |   10.9038 |  0.666496 |     2.61101 |      4.25614 |      7.5574  | 0.785817 |        0.022802 |  6.1235  |      0.044498 |    0.618824 |      0.734876 |             0.608567 |     0.304835 |

## Deltas from raw

| setting                |   best5_tm |   c1_rmsd |   c1_lddt |   sw_rmsd_9 |   sw_rmsd_15 |   sw_rmsd_31 |    bb_dev |   clash_per_res |    rg_err |   sharp_kinks |   angle_nll |   torsion_nll |   pair_like_fraction |   mean_drift |
|:-----------------------|-----------:|----------:|----------:|------------:|-------------:|-------------:|----------:|----------------:|----------:|--------------:|------------:|--------------:|---------------------:|-------------:|
| raw                    |   0        |  0        |  0        |    0        |     0        |     0        |  0        |        0        |  0        |      0        |    0        |      0        |              0       |     0        |
| simple_source_backbone |  -0.000537 | -0.000642 |  0.007581 |   -0.009801 |    -0.008612 |    -0.004104 | -0.202655 |       -0.024413 | -0.003351 |      0.019582 |    0.049047 |      0.034085 |             -0.00492 |     0.164081 |
| source_1p5             |  -0.001107 |  0.000122 |  0.00666  |   -0.040386 |    -0.017257 |    -0.005248 | -0.254453 |       -0.042319 | -0.050611 |     -0.004479 |   -0.201728 |     -0.102935 |             -0.00369 |     0.304835 |

## Paired target bootstrap

Positive deltas always mean the method in `setting` is better than `baseline`. C1-lDDT, sliding-window RMSD and TM are independent of the optimized Geometry-v2 loss. Angle/torsion NLL, clash and kink values are objective diagnostics only.

| baseline               | setting                | metric     |   n_targets |   mean_delta |   median_delta |    ci_low |   ci_high |   improved |   tied |   regressed |
|:-----------------------|:-----------------------|:-----------|------------:|-------------:|---------------:|----------:|----------:|-----------:|-------:|------------:|
| raw                    | simple_source_backbone | best5_tm   |          20 |    -0.000537 |      -0.000275 | -0.001369 |  0.000207 |          8 |      0 |          12 |
| raw                    | simple_source_backbone | c1_rmsd    |          20 |     0.000642 |       0.002702 | -0.007091 |  0.007842 |         12 |      0 |           8 |
| raw                    | simple_source_backbone | c1_lddt    |          20 |     0.007581 |       0.003325 |  0.00342  |  0.012358 |         17 |      0 |           3 |
| raw                    | simple_source_backbone | sw_rmsd_9  |          20 |     0.009801 |       0.014797 |  0.001439 |  0.01711  |         16 |      0 |           4 |
| raw                    | simple_source_backbone | sw_rmsd_15 |          20 |     0.008612 |       0.010399 |  0.000829 |  0.015565 |         16 |      0 |           4 |
| raw                    | simple_source_backbone | sw_rmsd_31 |          20 |     0.004104 |       0.005022 | -0.004059 |  0.011766 |         13 |      0 |           7 |
| raw                    | source_1p5             | best5_tm   |          20 |    -0.001107 |      -0.000375 | -0.002809 |  0.000418 |          8 |      0 |          12 |
| raw                    | source_1p5             | c1_rmsd    |          20 |    -0.000122 |       0.007785 | -0.019052 |  0.015333 |         12 |      0 |           8 |
| raw                    | source_1p5             | c1_lddt    |          20 |     0.00666  |       0.002773 |  0.000933 |  0.013093 |         12 |      0 |           8 |
| raw                    | source_1p5             | sw_rmsd_9  |          20 |     0.040386 |       0.03728  |  0.029354 |  0.05209  |         19 |      0 |           1 |
| raw                    | source_1p5             | sw_rmsd_15 |          20 |     0.017257 |       0.029497 |  0.000617 |  0.031336 |         15 |      0 |           5 |
| raw                    | source_1p5             | sw_rmsd_31 |          20 |     0.005248 |       0.011549 | -0.011112 |  0.019193 |         13 |      0 |           7 |
| simple_source_backbone | source_1p5             | best5_tm   |          20 |    -0.00057  |      -0.000385 | -0.001615 |  0.00043  |          5 |      0 |          15 |
| simple_source_backbone | source_1p5             | c1_rmsd    |          20 |    -0.000764 |       0.00583  | -0.014834 |  0.010612 |         12 |      0 |           8 |
| simple_source_backbone | source_1p5             | c1_lddt    |          20 |    -0.000921 |      -0.001013 | -0.003118 |  0.001327 |          8 |      0 |          12 |
| simple_source_backbone | source_1p5             | sw_rmsd_9  |          20 |     0.030585 |       0.02391  |  0.01964  |  0.044115 |         19 |      0 |           1 |
| simple_source_backbone | source_1p5             | sw_rmsd_15 |          20 |     0.008645 |       0.013915 | -0.003291 |  0.018831 |         16 |      0 |           4 |
| simple_source_backbone | source_1p5             | sw_rmsd_31 |          20 |     0.001144 |       0.009099 | -0.00952  |  0.010172 |         14 |      0 |           6 |

## Fixed-N 2×2 source-bank × Geometry factorial

Both banks contain exactly three candidates: 3 TBM versus 2 TBM + 1 DRfold2. This removes candidate-count advantage from the source-composition factor.

| bank           | geometry   |     mean |      std |   count |
|:---------------|:-----------|---------:|---------:|--------:|
| Hybrid (2T+1D) | off        | 0.581196 | 0.216359 |      20 |
| Hybrid (2T+1D) | on         | 0.578592 | 0.219916 |      20 |
| TBM (3T+0D)    | off        | 0.565183 | 0.233256 |      20 |
| TBM (3T+0D)    | on         | 0.559331 | 0.240606 |      20 |

| effect                   |   n_targets |   mean_delta |   median_delta |    ci_low |   ci_high |   improved |   tied |   regressed |
|:-------------------------|------------:|-------------:|---------------:|----------:|----------:|-----------:|-------:|------------:|
| hybrid_gain_geometry_off |          20 |     0.016012 |       0        | -0.002072 |  0.041466 |          7 |     10 |           3 |
| hybrid_gain_geometry_on  |          20 |     0.01926  |       0        | -0.000459 |  0.045383 |          7 |      9 |           4 |
| geometry_effect_tbm      |          20 |    -0.005852 |       0.00021  | -0.014781 |  0.000724 |         10 |      0 |          10 |
| geometry_effect_hybrid   |          20 |    -0.002604 |      -0.000455 | -0.005473 | -0.000278 |          7 |      0 |          13 |
| interaction              |          20 |     0.003248 |       0        | -0.00262  |  0.011143 |          6 |      9 |           5 |

The cache contains 3 TBM and 2 DRfold2 candidates per RNA. Therefore the requested full fixed-five 5T→0T allocation sweep is not identifiable from current artifacts. `candidate_allocation.csv` reports all honest fixed-size allocations supported at N=1,2,3, plus the production 3T+2D bank.

## Stratified paired effects

Candidates are first averaged inside each target/group cell; intervals then bootstrap RNA targets, not correlated candidate structures.

| group_type            | group       | metric      |   n_candidate_pairs |   n_targets |   mean_delta |   median_delta |    ci_low |   ci_high |   improved |   tied |   regressed |
|:----------------------|:------------|:------------|--------------------:|------------:|-------------:|---------------:|----------:|----------:|-----------:|-------:|------------:|
| source                | drfold2_e2e | c1_lddt     |                  40 |          20 |     0.005235 |       0.000517 | -0.002165 |  0.013847 |         10 |      0 |          10 |
| source                | drfold2_e2e | sw_rmsd_15  |                  40 |          20 |     0.01087  |       0.003424 | -0.001903 |  0.024892 |         11 |      0 |           9 |
| source                | drfold2_e2e | sharp_kinks |                  40 |          20 |     0.006991 |       0.006758 |  0.003731 |  0.010572 |         12 |      8 |           0 |
| source                | drfold2_e2e | angle_nll   |                  40 |          20 |     0.145222 |       0.144619 |  0.126413 |  0.162988 |         20 |      0 |           0 |
| source                | drfold2_e2e | torsion_nll |                  40 |          20 |     0.109221 |       0.102463 |  0.092023 |  0.127417 |         20 |      0 |           0 |
| source                | tbm         | c1_lddt     |                  60 |          20 |     0.00761  |       0.001431 |  0.001307 |  0.014217 |         12 |      0 |           8 |
| source                | tbm         | sw_rmsd_15  |                  60 |          20 |     0.021516 |       0.031094 | -0.002738 |  0.042679 |         17 |      0 |           3 |
| source                | tbm         | sharp_kinks |                  60 |          20 |     0.002805 |       0        |  0.001262 |  0.004608 |          8 |     12 |           0 |
| source                | tbm         | angle_nll   |                  60 |          20 |     0.239398 |       0.177163 |  0.169101 |  0.322566 |         20 |      0 |           0 |
| source                | tbm         | torsion_nll |                  60 |          20 |     0.098744 |       0.100442 |  0.084274 |  0.112801 |         20 |      0 |           0 |
| length_group          | short       | c1_lddt     |                  20 |           4 |     0.006123 |       0.001912 | -0.004326 |  0.019369 |          2 |      0 |           2 |
| length_group          | short       | sw_rmsd_15  |                  20 |           4 |    -0.006228 |       0.00625  | -0.063806 |  0.038872 |          2 |      0 |           2 |
| length_group          | short       | sharp_kinks |                  20 |           4 |     0.005527 |       0.006798 |  0.002083 |  0.008422 |          3 |      1 |           0 |
| length_group          | short       | angle_nll   |                  20 |           4 |     0.267733 |       0.256117 |  0.095249 |  0.440218 |          4 |      0 |           0 |
| length_group          | short       | torsion_nll |                  20 |           4 |     0.088593 |       0.093075 |  0.054863 |  0.122322 |          4 |      0 |           0 |
| length_group          | medium      | c1_lddt     |                  55 |          11 |     0.008949 |       0.00324  | -0.00029  |  0.019303 |          7 |      0 |           4 |
| length_group          | medium      | sw_rmsd_15  |                  55 |          11 |     0.020543 |       0.033252 |  0.003586 |  0.035145 |          9 |      0 |           2 |
| length_group          | medium      | sharp_kinks |                  55 |          11 |     0.004496 |       0.002899 |  0.002007 |  0.007165 |          7 |      4 |           0 |
| length_group          | medium      | angle_nll   |                  55 |          11 |     0.174686 |       0.160754 |  0.147506 |  0.2036   |         11 |      0 |           0 |
| length_group          | medium      | torsion_nll |                  55 |          11 |     0.108008 |       0.108385 |  0.094828 |  0.121087 |         11 |      0 |           0 |
| length_group          | long        | c1_lddt     |                  25 |           5 |     0.002056 |       0.000308 | -0.002887 |  0.006919 |          3 |      0 |           2 |
| length_group          | long        | sw_rmsd_15  |                  25 |           5 |     0.028817 |       0.036242 |  0.00906  |  0.047897 |          4 |      0 |           1 |
| length_group          | long        | sharp_kinks |                  25 |           5 |     0.003605 |       0.002667 |  0.000533 |  0.007396 |          3 |      2 |           0 |
| length_group          | long        | angle_nll   |                  25 |           5 |     0.208415 |       0.14123  |  0.123028 |  0.304692 |          5 |      0 |           0 |
| length_group          | long        | torsion_nll |                  25 |           5 |     0.103248 |       0.078715 |  0.077191 |  0.139093 |          5 |      0 |           0 |
| initial_quality_group | low         | c1_lddt     |                  34 |          11 |     0.016261 |       0.016418 |  0.0089   |  0.024089 |         10 |      0 |           1 |
| initial_quality_group | low         | sw_rmsd_15  |                  34 |          11 |     0.006305 |       0.019669 | -0.023126 |  0.035168 |          7 |      0 |           4 |
| initial_quality_group | low         | sharp_kinks |                  34 |          11 |     0.005034 |       0        |  0.000551 |  0.012006 |          4 |      7 |           0 |
| initial_quality_group | low         | angle_nll   |                  34 |          11 |     0.237113 |       0.25596  |  0.149537 |  0.323476 |         11 |      0 |           0 |
| initial_quality_group | low         | torsion_nll |                  34 |          11 |     0.117911 |       0.110383 |  0.101326 |  0.136328 |         11 |      0 |           0 |
| initial_quality_group | mid         | c1_lddt     |                  33 |          13 |     0.00388  |      -0.002251 | -0.003194 |  0.013347 |          5 |      0 |           8 |
| initial_quality_group | mid         | sw_rmsd_15  |                  33 |          13 |     0.026974 |       0.010589 |  0.010407 |  0.045965 |         10 |      0 |           3 |
| initial_quality_group | mid         | sharp_kinks |                  33 |          13 |     0.006174 |       0        |  0.001964 |  0.011017 |          6 |      7 |           0 |
| initial_quality_group | mid         | angle_nll   |                  33 |          13 |     0.191416 |       0.162467 |  0.136646 |  0.256079 |         13 |      0 |           0 |
| initial_quality_group | mid         | torsion_nll |                  33 |          13 |     0.114376 |       0.117871 |  0.08863  |  0.139003 |         13 |      0 |           0 |
| initial_quality_group | high        | c1_lddt     |                  33 |          10 |    -0.001788 |      -0.001576 | -0.005775 |  0.00242  |          3 |      0 |           7 |
| initial_quality_group | high        | sw_rmsd_15  |                  33 |          10 |     0.026077 |       0.030659 |  0.014115 |  0.035992 |          9 |      0 |           1 |
| initial_quality_group | high        | sharp_kinks |                  33 |          10 |     0.003526 |       0.003235 |  0.001751 |  0.005285 |          7 |      3 |           0 |
| initial_quality_group | high        | angle_nll   |                  33 |          10 |     0.147112 |       0.157214 |  0.132546 |  0.160405 |         10 |      0 |           0 |
| initial_quality_group | high        | torsion_nll |                  33 |          10 |     0.081639 |       0.073416 |  0.06748  |  0.097101 |         10 |      0 |           0 |
| confidence_group      | low         | c1_lddt     |                  34 |          16 |     0.006602 |       0.000517 | -0.0024   |  0.016553 |          8 |      0 |           8 |
| confidence_group      | low         | sw_rmsd_15  |                  34 |          16 |     0.01419  |       0.001438 | -0.001381 |  0.030745 |          8 |      0 |           8 |
| confidence_group      | low         | sharp_kinks |                  34 |          16 |     0.008233 |       0.007101 |  0.004783 |  0.012069 |         11 |      5 |           0 |
| confidence_group      | low         | angle_nll   |                  34 |          16 |     0.19704  |       0.152206 |  0.138063 |  0.273422 |         16 |      0 |           0 |
| confidence_group      | low         | torsion_nll |                  34 |          16 |     0.116041 |       0.116712 |  0.09435  |  0.138438 |         16 |      0 |           0 |
| confidence_group      | mid         | c1_lddt     |                  33 |          13 |     0.012573 |       0.010945 |  0.004213 |  0.021089 |         10 |      0 |           3 |
| confidence_group      | mid         | sw_rmsd_15  |                  33 |          13 |     0.02457  |       0.034715 | -0.008977 |  0.053749 |         10 |      0 |           3 |
| confidence_group      | mid         | sharp_kinks |                  33 |          13 |     0.001243 |       0        |  0        |  0.002733 |          3 |     10 |           0 |
| confidence_group      | mid         | angle_nll   |                  33 |          13 |     0.222516 |       0.193461 |  0.180939 |  0.268076 |         13 |      0 |           0 |
| confidence_group      | mid         | torsion_nll |                  33 |          13 |     0.101042 |       0.09869  |  0.086502 |  0.116293 |         13 |      0 |           0 |
| confidence_group      | high        | c1_lddt     |                  33 |          11 |    -0.001045 |      -0.004324 | -0.004326 |  0.002463 |          5 |      0 |           6 |
| confidence_group      | high        | sw_rmsd_15  |                  33 |          11 |     0.024149 |       0.02456  |  0.014235 |  0.034013 |         11 |      0 |           0 |
| confidence_group      | high        | sharp_kinks |                  33 |          11 |     0.003751 |       0.003788 |  0.001505 |  0.006272 |          6 |      5 |           0 |
| confidence_group      | high        | angle_nll   |                  33 |          11 |     0.13997  |       0.119826 |  0.099363 |  0.194844 |         11 |      0 |           0 |
| confidence_group      | high        | torsion_nll |                  33 |          11 |     0.081484 |       0.090769 |  0.063722 |  0.095993 |         11 |      0 |           0 |
| support_group         | low         | c1_lddt     |                  34 |          15 |     0.012764 |       0.001589 |  0.001867 |  0.026524 |          9 |      0 |           6 |
| support_group         | low         | sw_rmsd_15  |                  34 |          15 |     0.022794 |       0.039512 | -0.012126 |  0.053622 |         12 |      0 |           3 |
| support_group         | low         | sharp_kinks |                  34 |          15 |     0.003702 |       0        |  0.001162 |  0.006863 |          5 |     10 |           0 |
| support_group         | low         | angle_nll   |                  34 |          15 |     0.330873 |       0.231348 |  0.217669 |  0.470398 |         15 |      0 |           0 |
| support_group         | low         | torsion_nll |                  34 |          15 |     0.110063 |       0.119669 |  0.077887 |  0.137261 |         14 |      0 |           1 |
| support_group         | mid         | c1_lddt     |                  33 |          14 |     0.004427 |       0.00082  | -0.002121 |  0.013833 |          8 |      0 |           6 |
| support_group         | mid         | sw_rmsd_15  |                  33 |          14 |     0.014411 |       0.021706 | -0.001187 |  0.029526 |         10 |      0 |           4 |
| support_group         | mid         | sharp_kinks |                  33 |          14 |     0.00624  |       0.002976 |  0.002603 |  0.010274 |          7 |      7 |           0 |
| support_group         | mid         | angle_nll   |                  33 |          14 |     0.147912 |       0.148887 |  0.128221 |  0.168522 |         14 |      0 |           0 |
| support_group         | mid         | torsion_nll |                  33 |          14 |     0.104653 |       0.095528 |  0.086652 |  0.125555 |         14 |      0 |           0 |
| support_group         | high        | c1_lddt     |                  33 |           9 |     0.005712 |       0.001833 | -0.001366 |  0.013759 |          6 |      0 |           3 |
| support_group         | high        | sw_rmsd_15  |                  33 |           9 |     0.019399 |       0.017387 |  0.006501 |  0.031636 |          7 |      0 |           2 |
| support_group         | high        | sharp_kinks |                  33 |           9 |     0.004018 |       0.003788 |  0.001639 |  0.00684  |          6 |      3 |           0 |
| support_group         | high        | angle_nll   |                  33 |           9 |     0.12764  |       0.112879 |  0.106445 |  0.148648 |          9 |      0 |           0 |
| support_group         | high        | torsion_nll |                  33 |           9 |     0.092539 |       0.079736 |  0.073141 |  0.116906 |          9 |      0 |           0 |
