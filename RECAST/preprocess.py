"""
AnnData preprocessing for paired spatial × condition counterfactual CVAE.

- ``prepare_paired_anndata``: no SLAT — for simulated / already spot-matched pairs (e.g. DLPFC sim).
- ``prepare_paired_anndata_with_slat``: SLAT alignment + gene filter — for real paired slices (Fig.~3 style).

Raw counts must live in ``adata.X`` or ``adata.layers["count"]`` / ``layers["counts"]`` before calling.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple, Union

import anndata as ad
import numpy as np

try:
    import scanpy as sc
except ImportError as e:
    sc = None  # type: ignore
    _SCANPY_ERR = e
else:
    _SCANPY_ERR = None


def _require_scanpy() -> None:
    if sc is None:
        raise ImportError(
            "scanpy is required for normalize_total/log1p preprocessing. "
            "Install with `pip install scanpy`."
        ) from _SCANPY_ERR


def ensure_count_layer(adata: ad.AnnData, count_layer: str = "count") -> ad.AnnData:
    """Ensure ``layers[count_layer]`` holds raw counts (copy from ``X`` or common aliases)."""
    if count_layer in adata.layers:
        return adata
    if "counts" in adata.layers:
        adata.layers[count_layer] = adata.layers["counts"].copy()
        return adata
    adata.layers[count_layer] = adata.X.copy()
    return adata


def normalize_log_layer(
    adata: ad.AnnData,
    *,
    count_layer: str = "count",
    log_norm_layer: str = "log_norm",
    target_sum: float = 1e4,
) -> ad.AnnData:
    """
    Sets ``obs['library_size']``, then Scanpy normalize_total + log1p, stores result in ``layers[log_norm_layer]``.
    Leaves ``layers[count_layer]`` as raw counts; ``adata.X`` becomes log-normalized matrix (Scanpy convention).
    """
    _require_scanpy()
    ensure_count_layer(adata, count_layer)
    adata.obs["library_size"] = np.asarray(adata.layers[count_layer].sum(axis=1)).ravel()
    adata.X = adata.layers[count_layer].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers[log_norm_layer] = adata.X.copy()
    return adata


def joint_normalize_spatial(
    adata_A: ad.AnnData,
    adata_B: ad.AnnData,
    *,
    spatial_key: str = "spatial",
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Jointly scale both slides' coordinates to approximately [-1, 1] (same as DLPFC / tutorial scripts).
    """
    ca = np.asarray(adata_A.obsm[spatial_key], dtype=np.float32)
    cb = np.asarray(adata_B.obsm[spatial_key], dtype=np.float32)
    ca_c = ca - ca.mean(axis=0)
    cb_c = cb - cb.mean(axis=0)
    all_c = np.concatenate([ca_c, cb_c], axis=0)
    global_min = all_c.min(axis=0)
    global_max = all_c.max(axis=0)
    scale = np.maximum(global_max - global_min, 1e-6)
    adata_A.obsm[spatial_key] = 2.0 * (ca_c - global_min) / scale - 1.0
    adata_B.obsm[spatial_key] = 2.0 * (cb_c - global_min) / scale - 1.0
    return adata_A, adata_B


def prepare_paired_anndata(
    adata_A: ad.AnnData,
    adata_B: ad.AnnData,
    *,
    copy: bool = True,
    count_layer: str = "count",
    log_norm_layer: str = "log_norm",
    target_sum: float = 1e4,
    joint_normalize_spatial_coords: bool = True,
    spatial_key: str = "spatial",
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Full preprocessing for a matched pair (condition A / condition B) **without** SLAT.

    Use this for simulated data or when spots are already in 1:1 correspondence.
    """
    a = adata_A.copy() if copy else adata_A
    b = adata_B.copy() if copy else adata_B
    normalize_log_layer(a, count_layer=count_layer, log_norm_layer=log_norm_layer, target_sum=target_sum)
    normalize_log_layer(b, count_layer=count_layer, log_norm_layer=log_norm_layer, target_sum=target_sum)
    if joint_normalize_spatial_coords:
        joint_normalize_spatial(a, b, spatial_key=spatial_key)
    return a, b


def filter_genes_paired_mosta_style(
    adata_list: List[ad.AnnData],
    *,
    min_counts: int = 100,
    min_cells: int = 50,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Intersect genes and filter by pooled expression (matches ``spatialmeta_tutorial/Fig3.ipynb`` ``_filter_genes``).
    """
    if len(adata_list) != 2:
        raise ValueError("Expected exactly two AnnData objects.")
    common_genes = adata_list[0].var_names.intersection(adata_list[1].var_names)
    ad0 = adata_list[0][:, common_genes].copy()
    ad1 = adata_list[1][:, common_genes].copy()

    total_X = ad0.X + ad1.X
    gene_counts = np.asarray(total_X.sum(axis=0)).ravel()
    gene_cells = np.asarray((total_X > 0).sum(axis=0)).ravel()

    mask_expr = (gene_counts >= min_counts) & (gene_cells >= min_cells)
    mt_pattern = re.compile(r"^mt[-_]", re.IGNORECASE)
    mask_not_mt = ~ad0.var_names.str.match(mt_pattern)
    rp_pattern = re.compile(r"^RP[LS]", re.IGNORECASE)
    mask_not_rp = ~ad0.var_names.str.match(rp_pattern)

    final_mask = mask_expr & mask_not_mt & mask_not_rp
    return ad0[:, final_mask].copy(), ad1[:, final_mask].copy()


def _require_slat() -> None:
    try:
        import scSLAT  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "SLAT preprocessing requires the scSLAT package. "
            "Install it in the same environment used for the SpatialMETA tutorials, "
            "then retry (see https://github.com/zhanglabtools/scSLAT)."
        ) from e


def _normalize_log_for_slat_pca(adata: ad.AnnData, *, target_sum: float = 1e4) -> ad.AnnData:
    """In-place: set X from raw counts, normalize_total + log1p (for PCA / SLAT graph)."""
    _require_scanpy()
    ensure_count_layer(adata)
    adata.X = adata.layers["count"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def _ensure_pca_obsm(adata: ad.AnnData, n_comps: int = 50) -> None:
    _require_scanpy()
    if "X_pca" in adata.obsm:
        return
    n = min(int(n_comps), max(2, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=n)


def prepare_paired_anndata_with_slat(
    adata_ref_raw: ad.AnnData,
    adata_qry_raw: ad.AnnData,
    *,
    copy: bool = True,
    n_pca: int = 50,
    slat_feature: str = "pca",
    slat_device: Optional[Union[str, object]] = None,
    subsample_n_obs: Optional[int] = None,
    subsample_random_state: int = 0,
    filter_genes: bool = True,
    filter_min_counts: int = 100,
    filter_min_cells: int = 50,
    gene_filter_fn: Optional[Callable[[List[ad.AnnData]], Tuple[ad.AnnData, ad.AnnData]]] = None,
    count_layer: str = "count",
    log_norm_layer: str = "log_norm",
    target_sum: float = 1e4,
    store_spatial_raw: bool = True,
    spatial_key: str = "spatial",
    joint_normalize_spatial_coords: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData]:

    _require_scanpy()
    _require_slat()

    from .preprocess_slat import do_slat_pair, get_the_feature_new

    ad_ref_raw = adata_ref_raw.copy() if copy else adata_ref_raw
    ad_qry_raw = adata_qry_raw.copy() if copy else adata_qry_raw

    ensure_count_layer(ad_ref_raw, count_layer)
    ensure_count_layer(ad_qry_raw, count_layer)
    ad_ref_raw.X = ad_ref_raw.layers[count_layer].copy()
    ad_qry_raw.X = ad_qry_raw.layers[count_layer].copy()

    if subsample_n_obs is not None:
        sc.pp.subsample(ad_ref_raw, n_obs=subsample_n_obs, random_state=subsample_random_state)
        sc.pp.subsample(ad_qry_raw, n_obs=subsample_n_obs, random_state=subsample_random_state)

    ad_ref = ad_ref_raw.copy()
    ad_qry = ad_qry_raw.copy()
    ensure_count_layer(ad_ref, count_layer)
    ensure_count_layer(ad_qry, count_layer)

    _normalize_log_for_slat_pca(ad_ref, target_sum=target_sum)
    _normalize_log_for_slat_pca(ad_qry, target_sum=target_sum)

    _ensure_pca_obsm(ad_ref, n_comps=n_pca)
    _ensure_pca_obsm(ad_qry, n_comps=n_pca)

    _, best = do_slat_pair(ad_ref, ad_qry, feature=slat_feature, device=slat_device)
    matching = best[0]
    _, _, _, _, ad0_aligned, ad1_aligned = get_the_feature_new([ad_ref, ad_qry], [matching])

    if gene_filter_fn is not None:
        ad0_f, ad1_f = gene_filter_fn([ad0_aligned, ad1_aligned])
    elif filter_genes:
        ad0_f, ad1_f = filter_genes_paired_mosta_style(
            [ad0_aligned, ad1_aligned],
            min_counts=filter_min_counts,
            min_cells=filter_min_cells,
        )
    else:
        ad0_f, ad1_f = ad0_aligned.copy(), ad1_aligned.copy()

    ad0_raw = ad_ref_raw[ad0_f.obs_names, ad0_f.var_names].copy()
    ad1_raw = ad_qry_raw[ad1_f.obs_names, ad1_f.var_names].copy()
    ad0_raw.obsm = ad0_f.obsm.copy()
    ad1_raw.obsm = ad1_f.obsm.copy()

    import scipy.sparse as sp

    for adata in (ad0_raw, ad1_raw):
        if not sp.issparse(adata.X):
            adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
        else:
            adata.X = adata.X.astype(np.float32)
        adata.layers[count_layer] = adata.X.copy()
        if store_spatial_raw and spatial_key in adata.obsm:
            adata.obsm["spatial_raw"] = np.asarray(adata.obsm[spatial_key], dtype=np.float32).copy()

    normalize_log_layer(ad0_raw, count_layer=count_layer, log_norm_layer=log_norm_layer, target_sum=target_sum)
    normalize_log_layer(ad1_raw, count_layer=count_layer, log_norm_layer=log_norm_layer, target_sum=target_sum)

    if joint_normalize_spatial_coords:
        joint_normalize_spatial(ad0_raw, ad1_raw, spatial_key=spatial_key)

    return ad_ref, ad_qry, ad0_raw, ad1_raw
