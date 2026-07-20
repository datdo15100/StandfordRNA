# GeoFuse Geometry v2 temporal-safe priors

- Source: `train_v2` with `temporal_cutoff < 2022-05-27`
- Chains contributing local geometry: 3397
- Mean pair-like residue fraction: 0.6449
- Histogram bins: 72
- Runtime: 32.0 seconds

`pair_like` is an inference-available structural proxy based on complementary bases and candidate C1' distance. It is not a native secondary-structure label.

| context   |   angle_n |   angle_median_deg |   angle_p05_deg |   angle_p95_deg |   torsion_n |   torsion_median_deg |
|:----------|----------:|-------------------:|----------------:|----------------:|------------:|---------------------:|
| global    |   2162428 |            142.911 |          60.507 |         165.257 |     2157041 |              -84.233 |
| pair_like |   1542064 |            145.467 |          74.553 |         165.625 |     1895512 |              -92.018 |
| unpaired  |    620364 |            129.854 |          44.893 |         164.014 |      261529 |              -35.412 |
