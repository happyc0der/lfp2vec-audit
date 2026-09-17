# Fine-tune results (4class)

## `cross_lab_ibl_to_allen`

| run | band | groups | balanced accuracy | chance | ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|
| cross_lab_ibl_to_allen__all_target_groups__seed0 | full | 10 | 0.340 +/- 0.100 | 0.358 | 0.535 | 0.997 |

Stage 2 baselines on the same scheme, for comparison:

| features | balanced accuracy | ECE |
|---|---:|---:|
| geometry | 0.628 | 0.263 |
| amplitude | 0.509 | 0.181 |
| bandpower_full | 0.460 | 0.118 |
| w2v2_frozen | 0.377 | 0.478 |
| bandpower_clean | 0.366 | 0.132 |
