# V4 confirmatory results

## Evidence boundary

The frozen confirmatory evaluation contains 97 RNA targets in 86 regenerated MMseqs
sequence-similarity clusters. All native-blind outputs were generated and hashed before
the opening receipt was created. The final native labels were then opened once. No
method, candidate, metric, fallback rule or statistical procedure was changed after
opening. Generation and evaluation both completed with zero failures, and all 97
targets remain in every primary analysis.

Development results explain why components entered the frozen pipeline. The results in
this document determine whether the three main effects generalized. They are not used
to retune the system.

## Primary hypotheses

| Hypothesis | Confirmatory comparison | Comparator mean | Method mean | Mean paired effect | Cluster-bootstrap 95% CI | Holm-adjusted p | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| H1 | Retained Thesis 5T raw versus J-controlled 5T raw | 0.401 | 0.376 | -0.0255 | [-0.0471, -0.0072] | 0.995940 | Fail, negative |
| H2 | Thesis 3T+2D raw versus Thesis 5T raw | 0.376 | 0.410 | +0.0342 | [+0.0155, +0.0543] | 0.000380 | Pass |
| H3 | Selected Geometry versus Simple, SW-RMSD9 | 4.325 | 4.305 | +0.0205 Å reduction | [+0.0141, +0.0266] | 0.000030 | Pass local superiority |

H1 uses `TM_Thesis - TM_J-controlled`. H2 uses `TM_3T+2D - TM_5T`. H3 uses
`SW-RMSD9_Simple - SW-RMSD9_Geometry`, so a positive H3 effect means lower local
error. The unrounded values are used in every calculation.

### H1: retained Thesis TBM does not improve the controlled baseline

J-controlled 5T obtained mean best-of-five TM 0.401, whereas retained Thesis 5T
obtained 0.376. The mean paired difference was -0.0255, or a 6.358% decrease relative
to J-controlled. The interval was entirely below zero. At target level, the Thesis
bank improved 43 targets, tied 3 and regressed 51. H1 therefore fails, and the thesis
must not claim that its TBM group is more accurate than the controlled public
baseline.

This result reverses the small, inconclusive development estimate of +0.0057. The
predeclared length sensitivity indicates that the main negative effect is concentrated
in the 80-149 nt stratum, where the mean difference is -0.0453. The short 30-79 nt
stratum is approximately neutral. These strata are supporting analyses and do not
replace the all-target H1 result.

### H2: independent-source allocation improves the fixed five-candidate bank

Replacing two Thesis template slots with two direct DRfold2 candidates raised mean
best-of-five TM from 0.376 to 0.410. The mean paired gain was +0.0342, corresponding to
a 9.098% relative increase. The confidence interval remained above zero and the effect
passed Holm correction. The mixed bank improved 47 targets, tied 33 and regressed 17.

The mechanism results support complementarity rather than a claim that DRfold2 alone
is universally stronger. At fixed N=2, 2D scored 0.333 and 2T scored 0.348, while
1T+1D scored 0.392. For the five-candidate banks, mean pairwise self-TM decreased from
0.518 for 5T to 0.369 for 3T+2D, and the near-duplicate pair fraction decreased from
0.149 to 0.054. The mixed bank therefore covers more distinct structural hypotheses,
which improves the best-of-five objective even though the pretrained-only bank is not
the strongest source in isolation. Boltz was unavailable, so the broader
independent-source interpretation remains more defensible than a claim of unique
DRfold2 superiority.

### H3: Geometry improves SW-RMSD9 beyond Simple and passes the TM safeguard

Selected Geometry reduced mean candidate-level SW-RMSD9 from 4.325 Å to 4.305 Å.
The reduction was 0.0205 Å, or 0.473%, with a confidence interval entirely above zero.
Geometry improved 73 targets and regressed 24. It passed Holm-corrected local
superiority.

The bank-level `TM_Geometry - TM_Simple` effect was +0.00218, with cluster-bootstrap
95% CI [+0.00097, +0.00366]. The lower bound is above the preregistered noninferiority
margin of -0.005. Mean bank TM increased from 0.410 to 0.412, or 0.532%. H3 therefore
passes both required gates.

Supporting metrics are not uniformly favorable and must remain visible. Geometry
improved SW-RMSD15 by 0.0143 Å with a positive confidence interval. Global C1′ RMSD
changed by only 0.0028 Å and its interval crossed zero. Same-reference candidate TM
changed by +0.00018 and was inconclusive. C1′-lDDT decreased by 0.00521, with an
interval entirely below zero. The evidence therefore supports a narrow claim:
Geometry improves the preregistered local-window error and slightly raises bank-level
TM, but it does not improve every independent local metric.

## Frozen same-candidate refinement factorial

| Candidate bank | Raw TM | John fixed TM | Simple TM | Geometry TM |
|---|---:|---:|---:|---:|
| J-controlled | 0.401 | 0.393 | 0.399 | 0.402 |
| Thesis 3T+2D | 0.410 | 0.404 | 0.410 | 0.412 |

John fixed improves some local-window measurements but reduces bank TM by 0.0086 on
J-controlled candidates and by 0.0065 on Thesis candidates. The original-confidence
John sensitivity is less damaging than fixed confidence but remains below Raw in the
descriptive factorial. This illustrates why local correction cannot be judged without
the global TM safeguard.

On the Thesis bank, Simple is effectively neutral relative to Raw at bank level
(-0.00040, interval crossing zero). Geometry is +0.00178 above Raw with a positive
confidence interval. The primary attribution remains Geometry versus Simple because
that is the preregistered H3 comparison.

## Development-to-confirmatory interpretation

| Effect | Development mean effect | Confirmatory mean effect | Final interpretation |
|---|---:|---:|---|
| H1, Thesis TBM versus J-controlled | +0.0057, inconclusive | -0.0255 | Does not generalize; negative final evidence |
| H2, 3T+2D versus 5T | +0.0157, inconclusive | +0.0342 | Generalizes and passes |
| H3, Geometry versus Simple, SW-RMSD9 | +0.0234, passes development gate | +0.0205 | Generalizes and passes |
| H3 TM safeguard | +0.00079 | +0.00218 | Passes preservation margin; observed effect is positive |

The final scientific conclusion is therefore component-specific. The retained Thesis
TBM should be described as an unsuccessful replacement for J-controlled TBM on the
confirmatory population. The fixed-budget mixed-source candidate bank is the main
confirmed TM-score contribution. Geometry is a smaller confirmed refinement
contribution beyond Simple, with an explicit limitation from the unfavorable lDDT
result. Temporal filtering, distinct-PDB selection and frozen fallbacks remain
engineering safeguards rather than accuracy contributions unless their separate
development evidence supports a stronger statement.

## External benchmark boundary

The historical pre-V4 Kaggle submission scored public/private 0.62809/0.61390, versus
0.60084/0.60175 for the earlier TBM submission. That comparison changed candidate
composition and refinement together, so it is whole-pipeline historical evidence only.
It cannot attribute the gain to H1, H2 or H3 and is not the exact frozen V4 external
benchmark. The exact V4 Kaggle check remains a separate deployment step.

## Reproducibility artifacts

- Native-blind output freeze: `reports/thesis_v4/final_freeze/native_blind_output_freeze.json`
- Final opening receipt: `reports/thesis_v4/final_freeze/final_opening_receipt.json`
- Opening event: `reports/thesis_v4/confirmatory/native_label_opening_event.json`
- Primary inference: `reports/thesis_v4/confirmatory/primary_inference.json`
- Supporting inference: `reports/thesis_v4/confirmatory/supporting_inference.json`
- Raw per-target results and factorial tables: `reports/thesis_v4/confirmatory/`

