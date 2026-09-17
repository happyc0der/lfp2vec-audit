# Notebooks

- `inspect_chunks.ipynb` — *Stage 1.* Hand-inspection gate: plot a handful of real chunks per
  region per dataset and confirm the labels are plausible (ripples in CA1, theta in DG) before
  anything is trained on them. Added when real data lands.
- `colab_runner.ipynb` — *Stage 3.* Thin wrapper for Colab: clone, install, mount Drive, run an
  experiment YAML through the same CLI used locally.
