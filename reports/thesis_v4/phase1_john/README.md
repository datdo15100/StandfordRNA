# V4 Phase 1: public John reproduction audit

Status: **PARTIAL_DATASET_REPRODUCTION_NATIVE_BLIND_SMOKE_PASS**

- Required name: **reproduced publicly released John pipeline**.
- Native performance accessed in this audit: **No**.
- Public TBM capture hash: `c8eb61cbb97c85ac0a58c37ce88881596f27fac9a9b072837d159b0726ee9d9e`.
- Local John-style database: 7,155 unique sequences versus 18,815 coordinate groups shown in captured notebook output.
- Native-blind smoke target: `R1117v2`, length 30, five raw candidates, repeatable with frozen seed: **True**.
- Raw boundary: after transfer/gap completion and before John rule refiner/jitter.

This audit does not establish an exact winning-pipeline reproduction. It establishes a
hashable local port, makes data mismatch explicit, and removes process-dependent random
execution from the controlled baseline. `J-controlled` still requires the shared
DB-controlled snapshot and is tracked separately.
