# Raw-evaluation statistical correction

The coordinate scores, target-level TM tables, target-weighted means, wins/ties/losses, and cluster-bootstrap confidence intervals in this directory are valid.

The first execution of `evaluate-raw` enumerated target-level signs in the `exact_sign_flip` fields of `headline_effects.json` and `rq2_effects.json`. That treats the identical-sequence targets R1189 and R1190 as independent sign units and is not the intended dependence-aware test.

The canonical V5 inference is therefore `../V5_VERIFIED_EVIDENCE.json`. It recomputes the exact test using one shared sign per sequence cluster, preserves target weighting through the cluster sums, and enumerates all `2^11 = 2,048` cluster-sign assignments. No coordinates, target TM scores, method choices, or bootstrap settings were changed.

The earlier JSON files are retained only as an implementation audit trail; their `exact_sign_flip` fields must not be quoted.
