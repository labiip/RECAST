"""
Post-hoc drift scoring from counterfactual predictions (v6).

Requires a model exposing ``get_all_predictions(adata, which_enc=..., batch_size=...)``
as implemented by :class:`CounterfactualSpatialCVAE`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@runtime_checkable
class CounterfactualPredictor(Protocol):
    def get_all_predictions(
        self,
        adata: ad.AnnData,
        *,
        which_enc: str = "A",
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _library_size_column(adata: ad.AnnData) -> np.ndarray:
    if "library_size" in adata.obs:
        return np.asarray(adata.obs["library_size"], dtype=np.float64).reshape(-1, 1)
    if "lib_size" in adata.obs:
        return np.asarray(adata.obs["lib_size"], dtype=np.float64).reshape(-1, 1)
    if "count" in adata.layers:
        c = adata.layers["count"]
        s = np.asarray(c.sum(axis=1)).ravel()
        return s.reshape(-1, 1)
    x = adata.X
    if sp.issparse(x):
        s = np.asarray(x.sum(axis=1)).ravel()
    else:
        s = np.asarray(x.sum(axis=1)).ravel()
    return s.reshape(-1, 1)


def compute_drift_score(
    adata_A: ad.AnnData,
    adata_B: ad.AnnData,
    model: CounterfactualPredictor,
    eps: float = 1e-6,
    min_expr_logcpm: float = 0.5,
    top_n_perturb: int = 6775,
    n_grid: int = 20,
    min_spots_per_grid: int = 3,
    alpha: float = 0.3,
    blacklist: bool = True,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """
    Drift score from cross-condition predictions vs self-predictions (tutorial ``compute_drift_score_v6``).

    Parameters
    ----------
    adata_A, adata_B
        Paired AnnData; must share ``var_names``. Use the **same** objects passed to training
        (or copies with identical ``obs['library_size']`` / counts and ``obsm['spatial']``).
    model
        Trained model with ``get_all_predictions``.
    min_expr_logcpm
        Minimum mean log-CPM (from self-predictions) to retain a gene before perturb ranking.
    top_n_perturb
        Number of top perturbation genes to keep for spatial scoring and final table.
    n_grid
        Spatial grid size (``n_grid`` × ``n_grid``) for patch-wise consistency.
    alpha
        Weight on spatial rank vs perturbation rank in ``[0, 1]``.
    blacklist
        If True, drop Gm*, *-ps*, and listed sex-linked genes.
    batch_size
        Inference batch size for ``get_all_predictions``.
    """

    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _compute_cpm(X_count: np.ndarray, lib_size: np.ndarray, target: float = 1e4) -> np.ndarray:
        lib = lib_size.reshape(-1, 1) if lib_size.ndim == 1 else lib_size
        return np.log1p(X_count / lib * target)

    def _is_blacklisted(gene_name: str) -> bool:
        sex_chromosomes = [
            "Xist",
            "Tsix",
            "Eif2s3y",
            "Ddx3y",
            "Uty",
            "Kdm5d",
            "Zfy1",
            "Zfy2",
        ]
        return gene_name in sex_chromosomes or gene_name.startswith("Gm") or "-ps" in gene_name

    def _compute_spatial_drift_score(
        delta_log: np.ndarray,
        coords: np.ndarray,
        grid_n: int,
        min_spots: int,
    ) -> np.ndarray:
        xy = coords - coords.min(axis=0)
        xy = xy / (xy.max(axis=0) + eps) * (grid_n - eps)
        grid_id = xy[:, 0].astype(int) * grid_n + xy[:, 1].astype(int)

        valid_grids = [g for g in np.unique(grid_id) if (grid_id == g).sum() >= min_spots]
        if len(valid_grids) < 2:
            return np.zeros(delta_log.shape[1])

        grid_means = np.stack([delta_log[grid_id == g].mean(axis=0) for g in valid_grids], axis=0)

        grid_std = grid_means.std(axis=0)
        grid_mean_abs = np.abs(grid_means.mean(axis=0)) + eps
        inconsistency = grid_std / grid_mean_abs

        n_pos_grid = (grid_means > 0).sum(axis=0)
        n_neg_grid = (grid_means < 0).sum(axis=0)
        n_total_grid = len(valid_grids)

        p = n_pos_grid / n_total_grid + eps
        q = n_neg_grid / n_total_grid + eps
        sign_entropy = -(p * np.log(p) + q * np.log(q))

        score = 0.5 * (inconsistency / (inconsistency.max() + eps)) + 0.5 * (
            sign_entropy / (sign_entropy.max() + eps)
        )
        return score

    model.eval()

    XAA, XAB = model.get_all_predictions(adata_A, which_enc="A", batch_size=batch_size)
    XBA, XBB = model.get_all_predictions(adata_B, which_enc="B", batch_size=batch_size)

    XAA_np = _to_numpy(XAA)
    XAB_np = _to_numpy(XAB)
    XBB_np = _to_numpy(XBB)
    XBA_np = _to_numpy(XBA)

    lib_A = _library_size_column(adata_A)
    lib_B = _library_size_column(adata_B)

    XAA_log = _compute_cpm(XAA_np, lib_A)
    XAB_log = _compute_cpm(XAB_np, lib_A)
    XBB_log = _compute_cpm(XBB_np, lib_B)
    XBA_log = _compute_cpm(XBA_np, lib_B)

    all_genes = list(adata_A.var_names)
    g_n = len(all_genes)

    mean_expr_log = (XAA_log.mean(axis=0) + XBB_log.mean(axis=0)) / 2
    expr_mask = mean_expr_log >= min_expr_logcpm

    if blacklist:
        bl_mask = np.array([not _is_blacklisted(g) for g in all_genes])
        expr_mask = expr_mask & bl_mask

    candidate_idx = np.where(expr_mask)[0]
    print(f"[DriftScore] Total genes:         {g_n}")
    print(f"[DriftScore] After expr+blacklist: {len(candidate_idx)}")

    delta_A = np.abs(XAB_log - XAA_log)
    delta_B = np.abs(XBA_log - XBB_log)
    perturb_raw = (delta_A.mean(axis=0) + delta_B.mean(axis=0)) / 2

    cand_perturb = perturb_raw[candidate_idx]
    sorted_local = np.argsort(cand_perturb)[::-1][:top_n_perturb]
    top_idx = candidate_idx[sorted_local].copy()

    perturb_f = perturb_raw[top_idx]
    expr_log_f = mean_expr_log[top_idx]
    genes_f = [all_genes[i] for i in top_idx]

    x_a = adata_A.X
    x_b = adata_B.X
    x_a_dense = x_a.toarray() if sp.issparse(x_a) else np.asarray(x_a)
    x_b_dense = x_b.toarray() if sp.issparse(x_b) else np.asarray(x_b)
    mean_expr_raw_f = ((x_a_dense.mean(0) + x_b_dense.mean(0)) / 2)[top_idx]

    coords_A = np.asarray(adata_A.obsm["spatial"])
    coords_B = np.asarray(adata_B.obsm["spatial"])

    delta_a_f = (XAB_log - XAA_log)[:, top_idx]
    delta_b_f = (XBA_log - XBB_log)[:, top_idx]

    sp_a = _compute_spatial_drift_score(delta_a_f, coords_A, n_grid, min_spots_per_grid)
    sp_b = _compute_spatial_drift_score(delta_b_f, coords_B, n_grid, min_spots_per_grid)
    spatial_f = (sp_a + sp_b) / 2

    p_rank = np.argsort(np.argsort(perturb_f)).astype(float) / len(perturb_f)
    s_rank = np.argsort(np.argsort(spatial_f)).astype(float) / len(spatial_f)

    score = (1 - alpha) * p_rank + alpha * s_rank

    df = pd.DataFrame(
        {
            "gene": genes_f,
            "score": score,
            "perturb": perturb_f,
            "log_perturb": np.log1p(perturb_f),
            "spatial_spec": spatial_f,
            "spatial_A": sp_a,
            "spatial_B": sp_b,
            "mean_expr_log": expr_log_f,
            "mean_expr_raw": mean_expr_raw_f,
        }
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"[DriftScore] Final genes: {len(df)}")
    if len(df) > 0:
        top = df.iloc[0]
        print(
            f"[DriftScore] Top gene: {top['gene']}"
            f"  score={top['score']:.4f}"
            f"  perturb={top['perturb']:.4f}"
            f"  spatial={top['spatial_spec']:.4f}"
            f"  expr_log={top['mean_expr_log']:.3f}"
        )

    if "gene_type" in adata_A.var.columns:
        df["gene_type"] = df["gene"].map(
            lambda g: adata_A.var.loc[g, "gene_type"] if g in adata_A.var.index else "unknown"
        )
        print("\n[DriftScore] Statistics by gene type:")
        print(df.groupby("gene_type")[["perturb", "spatial_spec", "score"]].agg(["mean", "median"]))

    return df
