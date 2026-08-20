# V4 Phase 1: pretrained provenance audit

- Native performance accessed: **No**.
- DRfold2 clean source: commit `3667d6ac44c5b14cff94c36e0371224acb2cdae5`, tree `22af904928ace988971d2f1bde0988d7bb8a5f3e`.
- `cfg_97`: 20 checkpoint files, all individually hashed.
- RCLM checkpoint: `2b8d62e48d080f7fdff5eb03897cb91c214c53bcfddfa7171d1ae1a66bc055f5`.
- Local structural FASTA: 10,432 records, 3,414 unique sequences.
- Published structural rule: structures released before 2024.
- Published RCLM corpus: approximately 30 million RNAcentral Release 22 sequences,
  trained for 67,000 batches.
- Repository-provisional V4 pool audited: 172 targets.
- Exact structural-training sequence overlaps: 19.
- MMseqs 80% identity and 80% coverage overlaps: 75.
- Structural-overlap pass at this rule: 97.
- RCLM exact membership remains **UNVERIFIED** because the Release 22 training manifest
  is not present locally. DRfold2 is therefore not described as wholly time-safe.
- Boltz source commit and checkpoint are unavailable locally. Historical Boltz output
  is not admissible as V4 evidence until provenance is recovered.
