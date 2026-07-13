# Supervisor update — 14 July 2026

This folder is a self-contained meeting and laptop handoff package.

## Open first

- `supervisor_update.pptx` — ready-to-present deck.
- `speaker_notes_vi.md` — detailed Vietnamese speaking notes for every slide.
- `slides.md` — editable, reviewable slide source.
- `data_audit.md` — measured dataset inventory and EDA snapshot.
- `research_plan_review.md` — what GeoFuse adds, plus methodological corrections.
- `sources.md` — evidence map and external references.
- `laptop_handoff.md` — how to continue on the GTX 1650 / 16 GB Windows laptop.

## Regenerate

From the repository root in the `rna-fold` environment:

```bash
python -m pip install -r presentation/2026-07-14-supervisor-update/requirements.txt
python presentation/2026-07-14-supervisor-update/data_audit.py
python presentation/2026-07-14-supervisor-update/build_presentation.py
```

The deck deliberately separates:

1. measured temporal-safe local results;
2. deliberately leaked/oracle diagnostic results;
3. proposed GeoFuse-RNA work that has not yet been validated.

This distinction is essential: the 12 local CASP15 targets support controlled
ablation, while a Kaggle late submission is needed to measure the hidden private set.
