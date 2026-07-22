# Counterfactual Spatial CVAE

Paired spatial counterfactual conditional VAE for comparing two spatial transcriptomics conditions. This repository packages a cleaned refactor of the SpatialMETA tutorial workflow (`CVAE_coor_2`), with presets aligned to paper Figures 3–6.

## Repository layout

```
counterfactual_spatial_cvae/
├── counterfactual_spatial_cvae/   # Python package (model, training, drift score)
│   ├── config.py                  # Hyperparameter presets
│   ├── model.py                   # CounterfactualSpatialCVAE
│   ├── preprocess.py              # Paired AnnData preprocessing
│   ├── preprocess_slat.py         # SLAT alignment helpers (Fig. 4–6)
│   ├── drift_score.py             # Post-hoc drift scoring
│   └── ...
├── Figure3.ipynb                  # Simulated DLPFC slices + baseline training
├── Figure4.ipynb                  # MOSTA E12.5 vs E14.5 — classification & enrichment
├── Figure5.ipynb                  # MOSTA — counterfactual visualization & gene groups
├── Figure6.ipynb                  # Smaller paired real slices — classification
├── pyproject.toml
└── README.md
```

## Installation

```bash
cd counterfactual_spatial_cvae
pip install -e .
```

**Core dependencies:** Python ≥3.9, PyTorch, scanpy, anndata, scikit-learn, scipy, pandas, tqdm.

**Optional (Fig. 4–6):** [scSLAT](https://github.com/zhanglabtools/scSLAT) for cross-section alignment via `prepare_paired_anndata_with_slat`.

**GPU:** CUDA is recommended; notebooks fall back to CPU if unavailable.

## Quick start

```python
from RECAST import (
    CounterfactualSpatialCVAE,
    prepare_paired_anndata,
    preset_dlpfc_sim_baseline,
)

a, b = prepare_paired_anndata(adata_A, adata_B, copy=True)
preset = preset_dlpfc_sim_baseline()
model = CounterfactualSpatialCVAE.from_adata(a, b, model_config=preset.model, device="cuda:0")
history = model.fit(preset.training)
X_same, X_cf = model.get_all_predictions(a, which_enc="A", batch_size=1024)
```

## Configuration presets

Each notebook uses a dedicated preset from `config.py`:

| Notebook | Preset | Schedule | Use case |
|----------|--------|----------|----------|
| `Figure3.ipynb` | `preset_dlpfc_sim_baseline()` | RAMP | Simulated DLPFC A/B slices |
| `Figure4.ipynb` | `preset_fig3_recommended()` | STEP | MOSTA real data (E12.5 vs E14.5) |
| `Figure5.ipynb` | `preset_fig3_recommended()` | STEP | MOSTA real data — CF plots |
| `Figure6.ipynb` | `preset_fig6()` | STEP | Smaller paired real slices |

**Helper:** `recommend_batch_size(n_spots_a, n_spots_b, cap=1024)` — caps minibatch size to the smaller slide (used in Fig. 4–5).

### Preset summary

**`preset_dlpfc_sim_baseline`** (Figure 3 simulation)
- Architecture: latent=128, hidden=(1024, 512), cond_emb=8
- Training: 100 epochs, RAMP spatial schedule, spatial_weight=1.0

**`preset_fig3_recommended`** (Figures 4–5)
- Architecture: latent=128, hidden=(1024, 512), cond_emb=32, cf_beta=0.7
- Training: 200 epochs, STEP schedule, spatial_weight=0, KL warmup 20 epochs, CF from epoch 40

**`preset_fig6`** (Figure 6)
- Architecture: latent=128, hidden=(512, 256), cond_emb=32
- Training: 200 epochs, STEP schedule, spatial_weight=0, counterfactual_weight=40

## Reproducing figures

Run notebooks from the repository root so the package imports correctly:

```bash
jupyter lab Figure3.ipynb   # or Figure4/5/6
```

### Before running

1. **Edit `CONFIG` in each notebook** — set paths to your Visium / MOSTA data and output directories. Default paths point to local machine-specific locations and must be changed.
2. **Figure 3** requires DLPFC Visium data to synthesize paired slices, or pre-generated `adata_A` / `adata_B` h5ad files.
3. **Figures 4–6** require paired AnnData objects (`adata_A`, `adata_B` or `steady_adata`, `dss_adata`) and scSLAT for alignment.

### Drift scoring

After training, all notebooks call `model.compute_drift_score(...)`. Parameters are set inline in each notebook (e.g. `n_grid`, `alpha`, `top_n_perturb`); they are **not** part of `config.py` training presets.

```python
result = model.compute_drift_score(
    ad0_raw, ad1_raw,
    min_expr_logcpm=0.0,
    top_n_perturb=3000,
    n_grid=1000,
    alpha=0,
    blacklist=True,
    batch_size=1024,
)
```

## Data requirements

- Raw counts in `adata.X` or `layers["count"]` before preprocessing.
- After `prepare_paired_anndata` / `prepare_paired_anndata_with_slat`:
  - `layers["count"]`, `layers["log_norm"]`
  - `obs["library_size"]`
  - `obsm["spatial"]` (jointly normalized to ~[-1, 1] for paired slides)

## Package API

| Symbol | Description |
|--------|-------------|
| `CounterfactualSpatialCVAE` | Main model class |
| `prepare_paired_anndata` | Preprocess paired slides (Fig. 3) |
| `prepare_paired_anndata_with_slat` | SLAT alignment + preprocess (Fig. 4–6) |
| `compute_drift_score` | Standalone drift scoring function |
| `ModelConfig` / `TrainingConfig` | Dataclass configs for custom runs |

## License

MIT (see `pyproject.toml`).

## Citation

