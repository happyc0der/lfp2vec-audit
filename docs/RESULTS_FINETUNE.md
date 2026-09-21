# Fine-tune results (4class)

## `cross_lab_allen_to_ibl`

| run | band | groups | balanced accuracy | chance | ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|
| cross_lab_allen_to_ibl__all_target_groups__seed0 | full | 7 | 0.302 +/- 0.063 | 0.298 | 0.655 | 0.994 |

Stage 2 baselines on the same scheme, for comparison:

| features | balanced accuracy | ECE |
|---|---:|---:|
| amplitude | 0.401 | 0.190 |
| bandpower_full | 0.384 | 0.147 |
| bandpower_clean | 0.364 | 0.177 |
| w2v2_frozen | 0.304 | 0.631 |
| position | 0.298 | 0.439 |

## `cross_lab_ibl_to_allen`

| run | band | groups | balanced accuracy | chance | ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|
| cross_lab_ibl_to_allen__all_target_groups__seed0 | full | 10 | 0.340 +/- 0.100 | 0.358 | 0.535 | 0.997 |

Stage 2 baselines on the same scheme, for comparison:

| features | balanced accuracy | ECE |
|---|---:|---:|
| amplitude | 0.509 | 0.181 |
| bandpower_full | 0.460 | 0.118 |
| w2v2_frozen | 0.377 | 0.478 |
| bandpower_clean | 0.366 | 0.132 |
| position | 0.233 | 0.668 |

## `loso_ibl`

| run | band | groups | balanced accuracy | chance | ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|
| loso_ibl__0802ced5_probe00__seed0 | full | 1 | 0.631 | 0.250 | 0.187 | 0.997 |
| loso_ibl__0802ced5_probe01__seed0 | full | 1 | 0.697 | 0.333 | 0.272 | 0.985 |
| loso_ibl__0a018f12_probe00__seed0 | full | 1 | 0.819 | 0.333 | 0.101 | 0.993 |

Stage 2 baselines on the same scheme, for comparison:

| features | balanced accuracy | ECE |
|---|---:|---:|
| position | 0.773 | 0.135 |
| w2v2_frozen | 0.697 | 0.124 |
| amplitude | 0.560 | 0.119 |
| bandpower_full | 0.441 | 0.081 |
| bandpower_clean | 0.399 | 0.063 |

### `loso_ibl`: paired against the baselines on the same 3 sessions

| features | baseline | fine-tune | difference | fine-tune wins |
|---|---:|---:|---:|---:|
| amplitude | 0.593 | 0.716 | +0.122 | 3/3 |
| bandpower_clean | 0.429 | 0.716 | +0.287 | 3/3 |
| bandpower_full | 0.479 | 0.716 | +0.236 | 3/3 |
| position | 0.733 | 0.716 | -0.017 | 1/3 |
| w2v2_frozen | 0.709 | 0.716 | +0.007 | 2/3 |

## Leakage check

Permuted-label fine-tune: balanced accuracy 0.292 against a chance of 0.358.
