"""
Counterfactual spatial CVAE for paired spatial transcriptomics conditions.

Reproduction notebooks: ``Figure3.ipynb``–``Figure6.ipynb``.
"""

from .config import (
    ExperimentPreset,
    ModelConfig,
    TrainingConfig,
    TrainingSchedule,
    preset_dlpfc_sim_baseline,
    preset_fig3_recommended,
    preset_fig6,
    recommend_batch_size,
)
from .drift_score import compute_drift_score
from .model import CounterfactualSpatialCVAE
from .preprocess import (
    ensure_count_layer,
    filter_genes_paired_mosta_style,
    joint_normalize_spatial,
    normalize_log_layer,
    prepare_paired_anndata,
    prepare_paired_anndata_with_slat,
)

__all__ = [
    "compute_drift_score",
    "CounterfactualSpatialCVAE",
    "ModelConfig",
    "TrainingConfig",
    "TrainingSchedule",
    "ExperimentPreset",
    "preset_fig3_recommended",
    "preset_fig6",
    "preset_dlpfc_sim_baseline",
    "recommend_batch_size",
    "prepare_paired_anndata",
    "prepare_paired_anndata_with_slat",
    "filter_genes_paired_mosta_style",
    "normalize_log_layer",
    "ensure_count_layer",
    "joint_normalize_spatial",
]

__version__ = "0.1.0"
