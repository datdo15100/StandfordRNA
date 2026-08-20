# V4 mandatory checkpoint before final native opening

The V4 method and target manifest are frozen, but final native performance has not
been opened.

- Final manifest: 97 RNA targets in 86 MMseqs sequence-similarity clusters.
- Sampling: none; all targets passing the frozen structural-overlap and accessible
  exposure screens are retained.
- Native-blind outputs currently generated: 0/97 TBM, 0/97 raw hybrid, 0/97 refined.
- The runner exposes only `validate`, `build-tbm`, `assemble-raw`, `refine`, and
  `status`. It contains no evaluation command.
- `final_method_freeze.json` contains all method, data, checkpoint, prior, code, and
  statistics hashes.
- `native_blind_execution_plan.json` gives the exact commands for later generation.

RCLM corpus membership remains unavailable, so the final set passes the DRfold2
structural-training overlap screen but is not claimed to establish complete pretrained
model time-safety. The public John hybrid is also only partially reproducible and is
not described as the exact winning submission.

Stop here until the user reviews the consolidated report and explicitly authorizes the
single final-label opening.
