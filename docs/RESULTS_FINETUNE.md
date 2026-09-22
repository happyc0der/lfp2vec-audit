# Fine-tune results (4class)

## `cross_lab_allen_to_ibl`

| run | band | groups | balanced accuracy | chance | ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|
| cross_lab_allen_to_ibl__all_target_groups__seed0 | full | 14 | 0.302 +/- 0.061 | 0.298 | 0.655 | 0.994 |
| cross_lab_allen_to_ibl__all_target_groups__seed0__lp100 | <=100 Hz | 7 | 0.422 +/- 0.063 | 0.298 | 0.266 | 0.735 |
| cross_lab_allen_to_ibl__all_target_groups__seed0__whiten | full | 7 | 0.428 +/- 0.075 | 0.298 | 0.334 | 0.805 |
| cross_lab_allen_to_ibl__all_target_groups__seed1 | full | 7 | 0.363 +/- 0.091 | 0.298 | 0.299 | 0.999 |
| cross_lab_allen_to_ibl__all_target_groups__seed1__lp100 | <=100 Hz | 7 | 0.428 +/- 0.089 | 0.298 | 0.347 | 0.789 |
| cross_lab_allen_to_ibl__all_target_groups__seed2 | full | 7 | 0.295 +/- 0.059 | 0.298 | 0.645 | 0.997 |
| cross_lab_allen_to_ibl__all_target_groups__seed2__lp100 | <=100 Hz | 7 | 0.459 +/- 0.072 | 0.298 | 0.332 | 0.826 |

Stage 2 baselines on the same scheme, for comparison:

| features | balanced accuracy | ECE |
|---|---:|---:|
| amplitude | 0.401 | 0.190 |
| bandpower_full | 0.384 | 0.147 |
| bandpower_clean | 0.364 | 0.177 |
| spectrum_clean | 0.361 | 0.238 |
| w2v2_frozen | 0.304 | 0.631 |
| position | 0.298 | 0.439 |

## `cross_lab_ibl_to_allen`

| run | band | groups | balanced accuracy | chance | ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|
| cross_lab_ibl_to_allen__all_target_groups__seed0 | full | 20 | 0.340 +/- 0.098 | 0.358 | 0.535 | 0.997 |
| cross_lab_ibl_to_allen__all_target_groups__seed0__lp100 | <=100 Hz | 10 | 0.468 +/- 0.147 | 0.358 | 0.361 | 0.830 |
| cross_lab_ibl_to_allen__all_target_groups__seed0__whiten | full | 10 | 0.369 +/- 0.149 | 0.358 | 0.524 | 0.912 |
| cross_lab_ibl_to_allen__all_target_groups__seed1 | full | 10 | 0.361 +/- 0.090 | 0.358 | 0.511 | 0.999 |
| cross_lab_ibl_to_allen__all_target_groups__seed1__lp100 | <=100 Hz | 10 | 0.390 +/- 0.170 | 0.358 | 0.360 | 0.819 |
| cross_lab_ibl_to_allen__all_target_groups__seed2 | full | 10 | 0.347 +/- 0.083 | 0.358 | 0.522 | 0.999 |
| cross_lab_ibl_to_allen__all_target_groups__seed2__lp100 | <=100 Hz | 10 | 0.425 +/- 0.128 | 0.358 | 0.286 | 0.804 |

Stage 2 baselines on the same scheme, for comparison:

| features | balanced accuracy | ECE |
|---|---:|---:|
| amplitude | 0.509 | 0.181 |
| bandpower_full | 0.460 | 0.118 |
| w2v2_frozen | 0.377 | 0.478 |
| bandpower_clean | 0.366 | 0.132 |
| spectrum_clean | 0.330 | 0.284 |
| position | 0.233 | 0.668 |

## `loso_ibl`

| run | band | groups | balanced accuracy | chance | ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|
| loso_ibl__0802ced5_probe00__seed0 | full | 1 | 0.631 | 0.250 | 0.187 | 0.997 |
| loso_ibl__0802ced5_probe01__seed0 | full | 1 | 0.697 | 0.333 | 0.272 | 0.985 |
| loso_ibl__0a018f12_probe00__seed0 | full | 1 | 0.819 | 0.333 | 0.101 | 0.993 |
| loso_ibl__3638d102_probe01__seed0 | full | 1 | 0.766 | 0.333 | 0.077 | 0.999 |
| loso_ibl__54238fd6_probe00__seed0 | full | 1 | 0.716 | 0.250 | 0.133 | 0.995 |
| loso_ibl__5dcee0eb_probe00__seed0 | full | 1 | 0.834 | 0.333 | 0.083 | 0.998 |
| loso_ibl__d2832a38_probe00__seed0 | full | 1 | 0.694 | 0.250 | 0.082 | 0.994 |

Stage 2 baselines on the same scheme, for comparison:

| features | balanced accuracy | ECE |
|---|---:|---:|
| position | 0.773 | 0.135 |
| w2v2_frozen | 0.697 | 0.124 |
| amplitude | 0.560 | 0.119 |
| bandpower_full | 0.441 | 0.081 |
| spectrum_clean | 0.436 | 0.102 |
| bandpower_clean | 0.399 | 0.063 |

### `loso_ibl`: paired against the baselines on the same 7 sessions

| features | baseline | fine-tune | difference | fine-tune wins |
|---|---:|---:|---:|---:|
| amplitude | 0.560 | 0.737 | +0.176 | 7/7 |
| bandpower_clean | 0.399 | 0.737 | +0.338 | 7/7 |
| bandpower_full | 0.441 | 0.737 | +0.296 | 7/7 |
| position | 0.773 | 0.737 | -0.036 | 3/7 |
| spectrum_clean | 0.436 | 0.737 | +0.300 | 7/7 |
| w2v2_frozen | 0.697 | 0.737 | +0.040 | 5/7 |

## Leakage check

Permuted-label fine-tune: balanced accuracy 0.292 against a chance of 0.358.
