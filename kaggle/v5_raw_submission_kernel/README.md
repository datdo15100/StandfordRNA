# V5 frozen Raw scorer check

This private notebook publishes the byte-frozen `3T+2D Raw` candidate bank selected on CASP15 before its Kaggle score is observed.

The notebook verifies both the submission SHA-256 and the final-method-freeze SHA-256, checks the competition schema and row order, and copies the exact CSV to `/kaggle/working/submission.csv`. The submitted table is the frozen coordinate table reordered to match Kaggle's sample submission; a pre-score technical receipt proves exact equality after sorting by ID.

Because the downloaded Kaggle test sequence file is byte-identical to the CASP15 validation sequence file used for development, the late leaderboard result is a deployment/scorer compatibility check rather than an independent hidden-target evaluation.
