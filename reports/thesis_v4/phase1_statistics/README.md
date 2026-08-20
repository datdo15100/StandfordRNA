# V4 cluster-aware inference implementation

The exact preregistered procedure is implemented in
`src/rna3d/eval/v4_statistics.py` and has data-free unit tests.

- RNA target is the statistical unit.
- Regenerated MMseqs cluster is the dependence block.
- Point estimates are target-weighted.
- Confidence intervals use 10,000 cluster bootstrap replicates.
- One-sided raw p-values use 100,000 cluster sign flips.
- H1, H2 and H3 superiority use Holm step-down correction.
- H3 TM preservation is a separate confidence-interval gate.
- Non-finite target deltas are rejected, so a failed output cannot be silently removed.

No experiment or native performance was accessed while implementing or testing this
module. The tests use synthetic numeric arrays only.
