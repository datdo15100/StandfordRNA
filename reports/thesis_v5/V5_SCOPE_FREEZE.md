# V5 scientific scope freeze

## Scientific hierarchy

V5 uses the competition-native hierarchy:

1. training data and structural resources for priors, calibration, implementation and
   internal analysis;
2. the 12 CASP15 targets for transparent local validation and model development;
3. Kaggle hidden evaluation for one frozen complete-pipeline external benchmark.

CASP15 is not described as untouched confirmation. Its labels may be inspected during
development, but its small sample size constrains the strength of claims. Kaggle is used
only for the complete system and never to attribute a score difference to one component.

## Explicit exclusion of the V4 97-target cohort

The train-derived 97-target cohort and every analysis under
`reports/thesis_v4/confirmatory/` or `reports/thesis_v4/postconfirmatory_*` are excluded
from V5 component selection, V5 Results and the main thesis narrative. Their files are
preserved unchanged as project provenance. They cannot be cited to justify retaining or
dropping a V5 component.

## Scientific spine

The thesis asks how the major design choices in a competitive RNA 3D prediction system
affect best-of-five TM-score and candidate behaviour:

- **RQ1 - TBM:** under one CASP15 sandbox, how does the inherited public John TBM
  compare with the reconstructed thesis TBM, and which retrieval, ranking and
  reconstruction choices explain their behaviour?
- **RQ2 - source allocation:** under a fixed prediction budget, when does replacing
  template candidates with an independent pretrained source improve structural
  coverage and best-of-five TM-score?
- **RQ3 - refinement:** on identical raw candidates, how do the faithful John refiner,
  a controlled John variant, Simple smoothing and Geometry affect global and local
  accuracy metrics?

## Component decision policy

No component is deleted mechanically because of a small negative mean on 12 targets.
For every component V5 records:

1. origin of the idea;
2. intended problem;
3. exact algorithm;
4. expected effect stated before evaluation;
5. CASP15 experiment that tests the expectation.

Interpretation uses the observed effect, uncertainty, per-target results, availability,
diversity and interactions with other components. The exact reconstructed historical
TBM remains a complete contender even when an isolated ablation is weak.

## Historical reconstruction rule

The labels "historical V3 TBM" and "historical V3 hybrid" are internal provenance
labels. They may be used in audit artifacts, but the main thesis describes them by
their scientific composition rather than by project chronology. A pipeline is mapped
to a historical Kaggle score only when code, configuration and output provenance show
that it generated that submission.

## Freeze rule for external evaluation

After CASP15 component and complete-pipeline experiments, the final system must be
specified and hashed before its new Kaggle score is requested. The freeze includes the
template snapshot, pretrained weights, candidate allocation, refinement, fallback,
runtime limits, notebook, submission schema and code commit. A later Kaggle score is
whole-system external evidence only.

## Thesis format

The final manuscript is English-only and follows the immutable school template under
`reports/Template/`. It presents a scientific design story rather than project version
history:

`problem -> inherited baseline -> motivated components -> controlled CASP15 experiments
-> interactions -> complete-pipeline selection -> frozen Kaggle evaluation -> insights
and limitations`.

