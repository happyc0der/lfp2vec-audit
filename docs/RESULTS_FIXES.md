# Closing the cross-lab gap

Every configuration below uses **no labels from the target lab**. Post-processing is the
paper's own pipeline applied identically to every model, electrode position included.
Published values are read from Figure 2e and carry about ±0.01.

Raw accuracy is reported because that is the paper's cross-lab metric; balanced accuracy
because it is the honest one. `margin` is raw accuracy minus the target lab's
majority-class rate, the paper's definition of chance.

## IBL → Allen

Published, Figure 2e *(figure)*: raw 0.56 against majority 0.45, **margin +0.11**.

| configuration | stage | raw | margin | balanced | chance | cross-lab ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|---:|
| fine-tuned, full band (n=1 seeds) | raw | 0.423 | -0.033 | 0.242 | 0.250 | 0.560 | 0.997 |
| fine-tuned, full band (n=1 seeds) | temporal | 0.422 | -0.034 | 0.237 | 0.250 | 0.560 | 0.997 |
| fine-tuned, full band (n=1 seeds) | spatial | 0.428 | -0.028 | 0.240 | 0.250 | 0.560 | 0.997 |
| fine-tuned, filters matched (≤100 Hz) (n=2 seeds) | raw | 0.467 ± 0.043 | +0.011 ± 0.043 | 0.406 ± 0.069 | 0.250 | 0.351 | 0.825 |
| fine-tuned, filters matched (≤100 Hz) (n=2 seeds) | temporal | 0.519 ± 0.095 | +0.063 ± 0.095 | 0.447 ± 0.115 | 0.250 | 0.351 | 0.825 |
| fine-tuned, filters matched (≤100 Hz) (n=2 seeds) | spatial | 0.537 ± 0.092 | +0.081 ± 0.092 | 0.463 ± 0.131 | 0.250 | 0.351 | 0.825 |
| fine-tuned, per-probe whitening (n=1 seeds) | raw | 0.293 | -0.163 | 0.321 | 0.250 | 0.534 | 0.912 |
| fine-tuned, per-probe whitening (n=1 seeds) | temporal | 0.281 | -0.175 | 0.324 | 0.250 | 0.534 | 0.912 |
| fine-tuned, per-probe whitening (n=1 seeds) | spatial | 0.265 | -0.191 | 0.303 | 0.250 | 0.534 | 0.912 |
| untrained checkpoint + linear head, filters matched | raw | 0.454 | -0.001 | 0.401 | 0.250 | — | — |
| untrained checkpoint + linear head, filters matched | temporal | 0.540 | +0.084 | 0.471 | 0.250 | — | — |
| untrained checkpoint + linear head, filters matched | spatial | 0.542 | +0.086 | 0.479 | 0.250 | — | — |
| electrode position (control) | raw | 0.217 | -0.239 | 0.373 | 0.250 | — | — |
| electrode position (control) | temporal | 0.217 | -0.239 | 0.373 | 0.250 | — | — |
| electrode position (control) | spatial | 0.215 | -0.241 | 0.372 | 0.250 | — | — |
| band power (control) | raw | 0.520 | +0.064 | 0.379 | 0.250 | — | — |
| band power (control) | temporal | 0.596 | +0.141 | 0.424 | 0.250 | — | — |
| band power (control) | spatial | 0.622 | +0.167 | 0.436 | 0.250 | — | — |
| fine-tuned, full band (seed 0) + C: per-probe centering | — | 0.130 | -0.322 | 0.239 | 0.200 | 0.535 | 0.526 |
| *controls for the row above* | — | uncentred head 0.416 | | centred in-lab 0.647 | | | |
| fine-tuned, filters matched (seed 0) + C: per-probe centering | — | 0.486 | +0.034 | 0.355 | 0.200 | 0.389 | 0.494 |
| *controls for the row above* | — | uncentred head 0.459 | | centred in-lab 0.574 | | | |

## Allen → IBL

Published, Figure 2e *(figure)*: raw 0.49 against majority 0.37, **margin +0.12**.

| configuration | stage | raw | margin | balanced | chance | cross-lab ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|---:|
| fine-tuned, full band (n=1 seeds) | raw | 0.309 | -0.060 | 0.258 | 0.250 | 0.656 | 0.994 |
| fine-tuned, full band (n=1 seeds) | temporal | 0.307 | -0.063 | 0.256 | 0.250 | 0.656 | 0.994 |
| fine-tuned, full band (n=1 seeds) | spatial | 0.307 | -0.063 | 0.256 | 0.250 | 0.656 | 0.994 |
| fine-tuned, filters matched (≤100 Hz) (n=1 seeds) | raw | 0.485 | +0.116 | 0.370 | 0.250 | 0.258 | 0.735 |
| fine-tuned, filters matched (≤100 Hz) (n=1 seeds) | temporal | 0.495 | +0.125 | 0.361 | 0.250 | 0.258 | 0.735 |
| fine-tuned, filters matched (≤100 Hz) (n=1 seeds) | spatial | 0.493 | +0.124 | 0.360 | 0.250 | 0.258 | 0.735 |
| fine-tuned, per-probe whitening (n=1 seeds) | raw | 0.488 | +0.118 | 0.368 | 0.250 | 0.333 | 0.805 |
| fine-tuned, per-probe whitening (n=1 seeds) | temporal | 0.465 | +0.096 | 0.335 | 0.250 | 0.333 | 0.805 |
| fine-tuned, per-probe whitening (n=1 seeds) | spatial | 0.465 | +0.096 | 0.335 | 0.250 | 0.333 | 0.805 |
| untrained checkpoint + linear head, filters matched | raw | 0.491 | +0.121 | 0.389 | 0.250 | — | — |
| untrained checkpoint + linear head, filters matched | temporal | 0.502 | +0.133 | 0.367 | 0.250 | — | — |
| untrained checkpoint + linear head, filters matched | spatial | 0.498 | +0.129 | 0.364 | 0.250 | — | — |
| electrode position (control) | raw | 0.370 | +0.000 | 0.250 | 0.250 | — | — |
| electrode position (control) | temporal | 0.370 | +0.000 | 0.250 | 0.250 | — | — |
| electrode position (control) | spatial | 0.370 | +0.000 | 0.250 | 0.250 | — | — |
| band power (control) | raw | 0.412 | +0.042 | 0.333 | 0.250 | — | — |
| band power (control) | temporal | 0.403 | +0.033 | 0.332 | 0.250 | — | — |
| band power (control) | spatial | 0.398 | +0.029 | 0.329 | 0.250 | — | — |
| fine-tuned, filters matched (seed 0) + C: per-probe centering | — | 0.465 | +0.096 | 0.413 | 0.250 | 0.145 | 0.463 |
| *controls for the row above* | — | uncentred head 0.497 | | centred in-lab 0.638 | | | |
