# Baseline results (4class)

## `cross_lab_allen_to_ibl`

| features | model | folds | balanced accuracy | chance | above chance | ECE |
|---|---|---:|---:|---:|---:|---:|
| geometry | logreg | 7 | 0.521 ± 0.098 | 0.298 | +0.224 | 0.402 |
| amplitude | logreg | 7 | 0.401 ± 0.079 | 0.298 | +0.104 | 0.190 |
| bandpower_full | logreg | 7 | 0.384 ± 0.080 | 0.298 | +0.086 | 0.147 |
| bandpower_clean | logreg | 7 | 0.364 ± 0.056 | 0.298 | +0.067 | 0.177 |
| w2v2_frozen | logreg | 7 | 0.304 ± 0.057 | 0.298 | +0.007 | 0.631 |
| amplitude | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.074 |
| bandpower_clean | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.074 |
| bandpower_full | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.074 |
| geometry | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.074 |
| w2v2_frozen | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.074 |

Permutation controls on the same folds: mean balanced accuracy 0.293 against a mean chance of 0.298.

Paired across folds (Wilcoxon signed-rank):

| a | b | folds | median difference | a wins | p |
|---|---|---:|---:|---:|---:|
| amplitude | bandpower_clean | 7 | +0.047 | 5/7 | 0.219 |
| amplitude | bandpower_full | 7 | +0.023 | 4/7 | 0.578 |
| amplitude | geometry | 7 | -0.082 | 0/7 | 0.016 |
| amplitude | w2v2_frozen | 7 | +0.122 | 7/7 | 0.016 |
| bandpower_clean | bandpower_full | 7 | -0.022 | 2/7 | 0.375 |
| bandpower_clean | geometry | 7 | -0.138 | 0/7 | 0.016 |
| bandpower_clean | w2v2_frozen | 7 | +0.058 | 7/7 | 0.016 |
| bandpower_full | geometry | 7 | -0.133 | 0/7 | 0.016 |
| bandpower_full | w2v2_frozen | 7 | +0.080 | 6/7 | 0.031 |
| geometry | w2v2_frozen | 7 | +0.186 | 7/7 | 0.016 |

## `cross_lab_ibl_to_allen`

| features | model | folds | balanced accuracy | chance | above chance | ECE |
|---|---|---:|---:|---:|---:|---:|
| geometry | logreg | 10 | 0.628 ± 0.144 | 0.358 | +0.270 | 0.263 |
| amplitude | logreg | 10 | 0.509 ± 0.114 | 0.358 | +0.151 | 0.181 |
| bandpower_full | logreg | 10 | 0.460 ± 0.111 | 0.358 | +0.102 | 0.118 |
| w2v2_frozen | logreg | 10 | 0.377 ± 0.087 | 0.358 | +0.018 | 0.478 |
| bandpower_clean | logreg | 10 | 0.366 ± 0.116 | 0.358 | +0.008 | 0.132 |
| amplitude | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.108 |
| bandpower_clean | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.108 |
| bandpower_full | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.108 |
| geometry | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.108 |
| w2v2_frozen | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.108 |

Permutation controls on the same folds: mean balanced accuracy 0.346 against a mean chance of 0.358.

Paired across folds (Wilcoxon signed-rank):

| a | b | folds | median difference | a wins | p |
|---|---|---:|---:|---:|---:|
| amplitude | bandpower_clean | 10 | +0.101 | 10/10 | 0.002 |
| amplitude | bandpower_full | 10 | +0.045 | 8/10 | 0.064 |
| amplitude | geometry | 10 | -0.031 | 3/10 | 0.232 |
| amplitude | w2v2_frozen | 10 | +0.087 | 10/10 | 0.002 |
| bandpower_clean | bandpower_full | 10 | -0.084 | 1/10 | 0.004 |
| bandpower_clean | geometry | 10 | -0.168 | 1/10 | 0.004 |
| bandpower_clean | w2v2_frozen | 10 | +0.025 | 6/10 | 0.770 |
| bandpower_full | geometry | 10 | -0.076 | 1/10 | 0.037 |
| bandpower_full | w2v2_frozen | 10 | +0.094 | 7/10 | 0.049 |
| geometry | w2v2_frozen | 10 | +0.217 | 9/10 | 0.006 |

## `loso_allen`

| features | model | folds | balanced accuracy | chance | above chance | ECE |
|---|---|---:|---:|---:|---:|---:|
| geometry | logreg | 10 | 0.822 ± 0.159 | 0.358 | +0.464 | 0.134 |
| w2v2_frozen | logreg | 10 | 0.701 ± 0.125 | 0.358 | +0.343 | 0.111 |
| bandpower_full | logreg | 10 | 0.590 ± 0.118 | 0.358 | +0.231 | 0.102 |
| amplitude | logreg | 10 | 0.526 ± 0.089 | 0.358 | +0.168 | 0.152 |
| bandpower_clean | logreg | 10 | 0.480 ± 0.087 | 0.358 | +0.122 | 0.082 |
| amplitude | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.090 |
| bandpower_clean | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.090 |
| bandpower_full | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.090 |
| geometry | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.090 |
| w2v2_frozen | constant | 10 | 0.358 ± 0.079 | 0.358 | +0.000 | 0.090 |

Permutation controls on the same folds: mean balanced accuracy 0.356 against a mean chance of 0.358.

Paired across folds (Wilcoxon signed-rank):

| a | b | folds | median difference | a wins | p |
|---|---|---:|---:|---:|---:|
| amplitude | bandpower_clean | 10 | +0.055 | 7/10 | 0.084 |
| amplitude | bandpower_full | 10 | -0.051 | 2/10 | 0.010 |
| amplitude | geometry | 10 | -0.294 | 0/10 | 0.002 |
| amplitude | w2v2_frozen | 10 | -0.161 | 0/10 | 0.002 |
| bandpower_clean | bandpower_full | 10 | -0.122 | 0/10 | 0.002 |
| bandpower_clean | geometry | 10 | -0.351 | 0/10 | 0.002 |
| bandpower_clean | w2v2_frozen | 10 | -0.236 | 0/10 | 0.002 |
| bandpower_full | geometry | 10 | -0.243 | 0/10 | 0.002 |
| bandpower_full | w2v2_frozen | 10 | -0.109 | 1/10 | 0.004 |
| geometry | w2v2_frozen | 10 | +0.091 | 10/10 | 0.002 |

## `loso_ibl`

| features | model | folds | balanced accuracy | chance | above chance | ECE |
|---|---|---:|---:|---:|---:|---:|
| geometry | logreg | 7 | 0.817 ± 0.144 | 0.298 | +0.519 | 0.111 |
| w2v2_frozen | logreg | 7 | 0.697 ± 0.083 | 0.298 | +0.399 | 0.124 |
| amplitude | logreg | 7 | 0.560 ± 0.101 | 0.298 | +0.263 | 0.119 |
| bandpower_full | logreg | 7 | 0.441 ± 0.113 | 0.298 | +0.143 | 0.081 |
| bandpower_clean | logreg | 7 | 0.399 ± 0.087 | 0.298 | +0.102 | 0.063 |
| amplitude | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.060 |
| bandpower_clean | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.060 |
| bandpower_full | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.060 |
| geometry | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.060 |
| w2v2_frozen | constant | 7 | 0.298 ± 0.045 | 0.298 | +0.000 | 0.060 |

Permutation controls on the same folds: mean balanced accuracy 0.299 against a mean chance of 0.298.

Paired across folds (Wilcoxon signed-rank):

| a | b | folds | median difference | a wins | p |
|---|---|---:|---:|---:|---:|
| amplitude | bandpower_clean | 7 | +0.180 | 7/7 | 0.016 |
| amplitude | bandpower_full | 7 | +0.157 | 6/7 | 0.047 |
| amplitude | geometry | 7 | -0.283 | 0/7 | 0.016 |
| amplitude | w2v2_frozen | 7 | -0.120 | 0/7 | 0.016 |
| bandpower_clean | bandpower_full | 7 | -0.042 | 1/7 | 0.047 |
| bandpower_clean | geometry | 7 | -0.407 | 0/7 | 0.016 |
| bandpower_clean | w2v2_frozen | 7 | -0.275 | 0/7 | 0.016 |
| bandpower_full | geometry | 7 | -0.373 | 0/7 | 0.016 |
| bandpower_full | w2v2_frozen | 7 | -0.234 | 0/7 | 0.016 |
| geometry | w2v2_frozen | 7 | +0.089 | 7/7 | 0.016 |
