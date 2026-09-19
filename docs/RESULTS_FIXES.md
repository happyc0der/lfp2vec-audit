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
| reproduction (Stage 3) | raw | 0.420 | -0.032 | 0.193 | 0.200 | 0.560 | 0.997 |
| reproduction (Stage 3) | temporal | 0.418 | -0.034 | 0.189 | 0.200 | 0.560 | 0.997 |
| reproduction (Stage 3) | spatial | 0.424 | -0.028 | 0.192 | 0.200 | 0.560 | 0.997 |
| electrode position (control) | raw | 0.729 | +0.273 | 0.671 | 0.250 | — | — |
| electrode position (control) | temporal | 0.729 | +0.273 | 0.671 | 0.250 | — | — |
| electrode position (control) | spatial | 0.729 | +0.273 | 0.671 | 0.250 | — | — |

## Allen → IBL

Published, Figure 2e *(figure)*: raw 0.49 against majority 0.37, **margin +0.12**.

| configuration | stage | raw | margin | balanced | chance | cross-lab ECE | lab identity |
|---|---|---:|---:|---:|---:|---:|---:|
| reproduction (Stage 3) | raw | 0.309 | -0.060 | 0.258 | 0.250 | 0.656 | 0.994 |
| reproduction (Stage 3) | temporal | 0.307 | -0.063 | 0.256 | 0.250 | 0.656 | 0.994 |
| reproduction (Stage 3) | spatial | 0.307 | -0.063 | 0.256 | 0.250 | 0.656 | 0.994 |
| electrode position (control) | raw | 0.577 | +0.207 | 0.434 | 0.250 | — | — |
| electrode position (control) | temporal | 0.577 | +0.207 | 0.434 | 0.250 | — | — |
| electrode position (control) | spatial | 0.577 | +0.207 | 0.434 | 0.250 | — | — |
