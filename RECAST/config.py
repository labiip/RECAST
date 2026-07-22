"""Model and training configuration presets for Figure 3–6 reproduction notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class TrainingSchedule(str, Enum):
    """
    ``STEP`` — piecewise schedules: KL linear warmup, low early CF weight,
    spatial loss off then on.

    ``RAMP`` — smooth spatial ramp by epoch fractions, optional KL cosine warmup.
    """

    STEP = "step"
    RAMP = "ramp"


@dataclass
class ModelConfig:
    n_latent: int = 128
    hidden_stacks: Tuple[int, ...] = (1024, 512)
    cond_emb_dim: int = 32
    share_encoder: bool = True
    dropout_rate: float = 0.1
    h_dim: int = 256
    n_freq: int = 8
    film_hidden: int = 512
    n_spatial_neighbors: int = 6
    free_bits: float = 0.2
    cf_alpha: float = 1.0
    cf_beta: float = 0.5
    counterfactual_target_sum: float = 1e4


@dataclass
class TrainingConfig:
    schedule: TrainingSchedule = TrainingSchedule.STEP
    max_epoch: int = 200
    n_per_batch: int = 1024
    lr: float = 1e-4
    weight_decay: float = 1e-6
    recon_w: float = 10.0
    kl_weight: float = 0.3
    spatial_weight: float = 1e4
    counterfactual_weight: float = 20.0
    seed: Optional[int] = 42
    verbose: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    # STEP-only
    kl_warmup_epochs: int = 20
    cf_start_epoch: int = 40
    cf_pre_weight: float = 0.05
    spatial_start_epoch: int = 100
    counterfactual_clamp_max: Optional[float] = 5.0
    grad_clip_norm: float = 10.0
    # RAMP-only
    n_epochs_kl_warmup: int = 0
    spatial_warmup_frac: float = 0.3
    spatial_ramp_frac: float = 0.4
    replace_sampling: bool = False


@dataclass(frozen=True)
class ExperimentPreset:
    model: ModelConfig
    training: TrainingConfig
    description: str = ""


def preset_fig3_recommended() -> ExperimentPreset:
    """
    Fig. 4–5 real-data MOSTA (E12.5 vs E14.5): recon-first STEP, spatial off by default.

    Used by ``Figure4.ipynb`` and ``Figure5.ipynb``.
    """
    return ExperimentPreset(
        model=ModelConfig(
            n_latent=128,
            hidden_stacks=(1024, 512),
            cond_emb_dim=32,
            cf_alpha=1.0,
            cf_beta=0.7,
        ),
        training=TrainingConfig(
            schedule=TrainingSchedule.STEP,
            max_epoch=200,
            n_per_batch=1024,
            lr=1e-4,
            recon_w=10.0,
            kl_weight=0.2,
            spatial_weight=0.0,
            counterfactual_weight=20.0,
            kl_warmup_epochs=20,
            cf_start_epoch=40,
            cf_pre_weight=0.05,
            spatial_start_epoch=120,
            counterfactual_clamp_max=5.0,
            grad_clip_norm=10.0,
            seed=42,
        ),
        description="Fig. 4–5 real ST — STEP, CF epoch≥40, spatial=0.",
    )


def preset_fig6() -> ExperimentPreset:
    """Fig. 6 real-data paired slices (``TrainingSchedule.STEP``, lighter spatial)."""
    return ExperimentPreset(
        model=ModelConfig(
            n_latent=128,
            hidden_stacks=(512, 256),
            cond_emb_dim=32,
        ),
        training=TrainingConfig(
            schedule=TrainingSchedule.STEP,
            max_epoch=200,
            n_per_batch=1024,
            spatial_start_epoch=60,
            lr=1e-4,
            recon_w=10.0,
            kl_weight=1.0,
            spatial_weight=0.0,
            counterfactual_weight=40.0,
        ),
        description="Fig. 6 real ST — step schedule.",
    )


def preset_dlpfc_sim_baseline() -> ExperimentPreset:
    """Simulated DLPFC baseline in ``Figure3.ipynb`` (``TrainingSchedule.RAMP``)."""
    return ExperimentPreset(
        model=ModelConfig(
            n_latent=128,
            hidden_stacks=(1024, 512),
            cond_emb_dim=8,
        ),
        training=TrainingConfig(
            schedule=TrainingSchedule.RAMP,
            max_epoch=100,
            n_per_batch=1024,
            lr=1e-4,
            recon_w=10.0,
            kl_weight=1.0,
            spatial_weight=1.0,
            counterfactual_weight=20.0,
            n_epochs_kl_warmup=0,
            spatial_warmup_frac=0.3,
            spatial_ramp_frac=0.4,
            counterfactual_clamp_max=None,
            grad_clip_norm=5.0,
            replace_sampling=False,
        ),
        description="Fig. 3 DLPFC simulation — ramp schedule.",
    )


def recommend_batch_size(n_spots_a: int, n_spots_b: int, cap: int = 1024) -> int:
    """Cap per-epoch minibatch by the smaller slide."""
    return min(cap, n_spots_a, n_spots_b)
