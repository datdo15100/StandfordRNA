# V4 Phase 1: P0-production reproduction

Status: **PASS**

- Target manifest: 3,397 train-V2 RNAs with `temporal_cutoff < 2022-05-27`.
- Prediction/native performance accessed: **No**.
- Distance/Rg prior match at tolerance `1e-10`: **True**; maximum absolute error `0`.
- Angle/torsion prior match at tolerance `1e-10`, excluding runtime metadata: **True**; maximum absolute error `0`.
- Production artifacts were read-only. Rebuilt artifacts are stored under `data/processed/v4_p0_rebuild`.
- Geometry production configuration is serialized in `p0_reproduction_audit.json`, including `steps=300` and `w_source=3.0`.

This audit establishes whether `P0-production` can be reconstructed from its declared
3,397-RNA input and current frozen code. It does not test whether P0 improves an RNA
prediction and does not authorize any change to the prior.
