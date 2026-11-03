Lineage inferred without metadata. Edge scores on test-split pairs; downstream is hybrid top-10 with the `promote` policy on test-split queries. *Gap closed* is the share of the current@1 gap between no lineage and oracle lineage.

**declared**: 120575 candidate pairs, candidate recall 0.903

| method | edge P | edge R | edge F1 | title queries: current@1 (gap closed) | questions: current@1 (gap closed) |
|---|---|---|---|---|---|
| no_lineage | – | – | – | 0.248 (0%) | 0.204 (0%) |
| heuristic | 0.325 | 0.541 | 0.406 | 0.612 (54%) | 0.694 (69%) |
| logreg | 0.775 | 0.578 | 0.663 | 0.660 (61%) | 0.694 (69%) |
| oracle | – | – | – | 0.925 (100%) | 0.918 (100%) |

**masked**: 120319 candidate pairs, candidate recall 0.898

| method | edge P | edge R | edge F1 | title queries: current@1 (gap closed) | questions: current@1 (gap closed) |
|---|---|---|---|---|---|
| no_lineage | – | – | – | 0.248 (0%) | 0.204 (0%) |
| heuristic | 0.288 | 0.449 | 0.351 | 0.543 (44%) | 0.592 (54%) |
| logreg | 0.593 | 0.481 | 0.531 | 0.594 (51%) | 0.551 (49%) |
| oracle | – | – | – | 0.925 (100%) | 0.918 (100%) |
