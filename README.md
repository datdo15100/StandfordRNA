# Stanford RNA 3D Folding

A temporal-safe RNA 3D structure prediction pipeline built for the Stanford RNA
3D Folding Kaggle competition and an accompanying thesis study. The project combines
template-based modelling (TBM), composite sequence/structure search, de novo fallback
candidates, and optional geometry-aware refinement.

The repository contains source code and experiment reports only. Competition data,
derived caches, pretrained weights, submissions, and external binaries are deliberately
excluded from Git.

## Competition introduction

### Overview

If you sat down to complete a puzzle without knowing what it should look like, you’d
have to rely on patterns and logic to piece it together. In the same way, predicting
Ribonucleic acid (RNA)’s 3D structure involves using only its sequence to figure out how
it folds into the structures that define its function.

In this competition, you’ll develop machine learning models to predict an RNA
molecule’s 3D structure from its sequence. The goal is to improve our understanding of
biological processes and drive new advancements in medicine and biotechnology.

### Description

RNA is vital to life’s most essential processes, but despite its significance,
predicting its 3D structure is still difficult. Deep learning breakthroughs like
AlphaFold have transformed protein structure prediction, but progress with RNA has been
much slower due to limited data and evaluation methods.

This competition builds on recent advances, like the deep learning foundation model
RibonanzaNet, which emerged from a prior Kaggle competition. Now, you’ll take on the
next challenge—predicting RNA’s full 3D structure.

Your work could push RNA-based medicine forward, making treatments like cancer
immunotherapies and CRISPR gene editing more accessible and effective. More
fundamentally, your work may be the key step in illuminating the folds and functions of
natural RNA molecules, which have been called the “dark matter of biology.”

This competition is made possible through a worldwide collaborative effort including
the organizers, experimental RNA structural biologists, and predictors of the CASP16
and RNA-Puzzles competitions; Howard Hughes Medical Institute; the Institute of Protein
Design; and Stanford University School of Medicine.

## Highlights

- Hard temporal filtering prevents templates released after a target cutoff from leaking
  into validation.
- Composite template search raised temporal-safe CASP15 best-of-5 TM-score from `0.2117`
  to `0.3072` in the recorded ablation.
- Five-candidate generation combines template candidates with de novo hedges.
- US-align scoring, leakage demonstrations, refiner ablations, and reproducible reports
  are included.
- Gradient refinement improves clash and backbone-spacing objectives, but the experiments
  also document its sharp-kink trade-off rather than claiming a uniform geometry gain.

See [the composite-search ablation](reports/thesis_notes/composite_ablation.md) and
[the refinement analysis](reports/thesis_notes/refine_ablation.md) for the measured
results and caveats.

## Requirements

- Linux or WSL2
- Conda
- Python 3.12
- MMseqs2
- US-align
- The Stanford RNA 3D Folding competition dataset

A CUDA-capable GPU is useful for refinement and pretrained-model experiments, but most
template database and scoring utilities can run on CPU.

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate rna-fold
```

Point the project at the competition data directory:

```bash
export RNA3D_DATA=/path/to/stanford-rna-3d-folding
```

Build US-align at `external/binaries/USalign`:

```bash
git clone --depth 1 https://github.com/pylelab/USalign /tmp/USalign
mkdir -p external/binaries
g++ -O3 -ffast-math -o external/binaries/USalign /tmp/USalign/USalign.cpp
```

For machine migration, data placement, CUDA notes, and artifact restoration, follow
[SETUP.md](SETUP.md).

## Data layout

By default, `RNA3D_DATA` should point to a directory containing the Kaggle files:

```text
stanford-rna-3d-folding/
├── train_sequences.csv
├── train_labels.csv
├── validation_sequences.csv
├── validation_labels.csv
├── test_sequences.csv
├── MSA/
├── MSA_v2/
└── PDB_RNA/
```

Paths and the CASP15 temporal cutoff are defined in `configs/paths.yaml`. Set
`RNA3D_CACHE` to move rebuildable caches outside the repository.

## Reproduce the pipeline

Rebuild priors and template-search artifacts:

```bash
bash scripts/rebuild_artifacts.sh
```

Run the scoring sanity checks and a short end-to-end evaluation:

```bash
python scripts/run_phase1_scoring.py
python scripts/run_eval.py --targets 3
```

Run the recorded headline experiments:

```bash
python scripts/reproduce_top1.py
python scripts/run_composite_ablation.py
python scripts/run_refine_ablation.py
```

Generate a competition submission after the required artifacts exist:

```bash
python scripts/make_submission.py
```

Generated tables, figures, caches, and submissions are Git-ignored. The checked-in
reports capture the results used in the thesis narrative.

## Tests

The lightweight checks do not require the competition dataset:

```bash
python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/rna3d-pycache python -m compileall -q \
  src scripts kaggle tests test.py
```

Full pipeline validation additionally requires the dataset, MMseqs2, and US-align.

## Repository map

```text
src/rna3d/              Core parsing, search, TBM, geometry, refinement, and evaluation
scripts/                Rebuild, evaluation, ablation, and submission entry points
kaggle/                 Offline Kaggle inference pipeline
configs/                Central path and cutoff configuration
reports/thesis_notes/   Experiment results and methodology notes
utilities/              Notebook exports and reference material (not all are executable)
```

Some files under `utilities/` intentionally preserve exploratory notebook cells and
captured output for reading. They are not included in syntax checks or CI.

Additional project context is in [TOPSOLUTION.md](TOPSOLUTION.md), [PLAN.md](PLAN.md),
and [LOG.md](LOG.md).

## Reproducibility and leakage policy

CASP15 validation uses templates released strictly before `2022-05-27`. Candidate
construction also excludes the target's own PDB identifier. Unknown release dates are
treated as invalid. Keep this gate enabled when comparing methods; leaked-template
results are useful only as explicitly labelled demonstrations.

## Current limitations

- No-template and very long targets remain difficult.
- Full reproduction depends on competition data that cannot be distributed here.
- The current gradient refiner reduces clashes and adjacent-distance error but can
  increase sharp pseudo-bond-angle kinks; see the refinement ablation before using it
  as a general physical-validity claim.
