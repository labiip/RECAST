from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm.auto import tqdm

from .config import ModelConfig, TrainingConfig, TrainingSchedule
from ._loss import zinb_reconstruction_loss


def _to_dense_rowslice(matrix, idx: np.ndarray) -> np.ndarray:
    sub = matrix[idx]
    return sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)


class _StackEncoder(nn.Module):
    """MLP encoder matching the original SAE(encode_only) layout: Linear → LayerNorm → ReLU → Dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_stacks: Tuple[int, ...],
        dropout_rate: float,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_stacks:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                activation(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            ]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CounterfactualSpatialCVAE(nn.Module):
    """
    Paired spatial counterfactual conditional VAE (two conditions, shared or split encoders).

    Expects AnnData objects after :func:`counterfactual_spatial_cvae.preprocess.prepare_paired_anndata`:
    ``layers['count']``, ``layers['log_norm']``, ``obs['library_size']``, ``obsm['spatial']``.
    """

    def __init__(
        self,
        adata_A: ad.AnnData,
        adata_B: ad.AnnData,
        config: Optional[ModelConfig] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.cfg = config or ModelConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.X_A_counts = adata_A.layers["count"]
        self.X_A_log = adata_A.layers["log_norm"]
        self.X_B_counts = adata_B.layers["count"]
        self.X_B_log = adata_B.layers["log_norm"]

        self.libA_global = torch.tensor(
            adata_A.obs["library_size"].values, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        self.libB_global = torch.tensor(
            adata_B.obs["library_size"].values, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)

        self.n_genes = self.X_A_counts.shape[1]
        self.nA = self.X_A_counts.shape[0]
        self.nB = self.X_B_counts.shape[0]
        self._ptrA = 0
        self._ptrB = 0

        self.coords_A = torch.as_tensor(adata_A.obsm["spatial"], dtype=torch.float32, device=self.device)
        self.coords_B = torch.as_tensor(adata_B.obsm["spatial"], dtype=torch.float32, device=self.device)

        hs = list(self.cfg.hidden_stacks)
        self.hidden_stacks = hs
        self.n_hidden = hs[-1]
        self.n_latent = self.cfg.n_latent

        enc = _StackEncoder(
            self.n_genes, tuple(hs), self.cfg.dropout_rate, nn.ReLU
        )
        self.encoder_A = enc
        if self.cfg.share_encoder:
            self.encoder_B = self.encoder_A
        else:
            self.encoder_B = _StackEncoder(
                self.n_genes, tuple(hs), self.cfg.dropout_rate, nn.ReLU
            )

        self.z_mean_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_fc = nn.Linear(self.n_hidden, self.n_latent)

        self.h_dim = self.cfg.h_dim
        self.cond_emb = nn.Embedding(2, self.cfg.cond_emb_dim)
        self.n_freq = self.cfg.n_freq
        pos_dim = 2 + 2 * 2 * self.n_freq
        self.pos_backbone = nn.Sequential(
            nn.Linear(pos_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )
        self.film_generator = nn.Sequential(
            nn.Linear(self.n_latent + self.cfg.cond_emb_dim + self.h_dim, self.cfg.film_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.cfg.film_hidden, self.h_dim * 2),
        )

        decoder_stacks = hs[::-1]
        dec_layers: List[nn.Module] = []
        cur = self.h_dim
        for h_dim in decoder_stacks:
            dec_layers += [
                nn.Linear(cur, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ]
            cur = h_dim
        self.decoder_backbone = nn.Sequential(*dec_layers)
        final_h = decoder_stacks[-1]
        self.px_rna_scale_decoder = nn.Sequential(nn.Linear(final_h, self.n_genes), nn.Softplus())
        self.px_rna_rate_decoder = nn.Linear(final_h, self.n_genes)
        self.px_rna_dropout_decoder = nn.Linear(final_h, self.n_genes)

        self.to(self.device)
        self.trained_state_dict: Optional[Dict[str, Any]] = None

    @classmethod
    def from_adata(
        cls,
        adata_A: ad.AnnData,
        adata_B: ad.AnnData,
        *,
        model_config: Optional[ModelConfig] = None,
        device: str | torch.device = "cpu",
    ) -> CounterfactualSpatialCVAE:
        required = {"count", "log_norm"}
        for name, adata in ("A", adata_A), ("B", adata_B):
            missing = required - set(adata.layers.keys())
            if missing:
                raise KeyError(f"adata_{name} missing layers: {missing}")
            if "library_size" not in adata.obs:
                raise KeyError(f"adata_{name} missing obs['library_size']")
            if "spatial" not in adata.obsm:
                raise KeyError(f"adata_{name} missing obsm['spatial']")
        return cls(adata_A, adata_B, config=model_config, device=device)

    def _encode_batch(self, X_log: torch.Tensor, which: str) -> torch.Tensor:
        enc = self.encoder_A if which == "A" else self.encoder_B
        return enc(X_log)

    def _z_from_q(self, q: torch.Tensor, eps: float = 1e-4) -> Dict[str, torch.Tensor]:
        mu = self.z_mean_fc(q)
        var = torch.exp(self.z_var_fc(q)) + eps
        if self.training:
            z = Normal(mu, var.sqrt()).rsample()
        else:
            z = mu
        return {"mu": mu, "var": var, "z": z}

    def _get_sinusoidal_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        freq_bands = 2.0 ** torch.linspace(
            0, self.n_freq - 1, self.n_freq, device=coords.device
        )
        out = [coords]
        for freq in freq_bands:
            out.append(torch.sin(coords * freq * math.pi))
            out.append(torch.cos(coords * freq * math.pi))
        return torch.cat(out, dim=-1)

    def _decode_z(
        self,
        z: torch.Tensor,
        cond_idx: torch.Tensor,
        coords: torch.Tensor,
        lib_size: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pos_enc = self._get_sinusoidal_encoding(coords)
        canvas = self.pos_backbone(pos_enc)
        c_emb = self.cond_emb(cond_idx)
        film_input = torch.cat([z, c_emb, canvas], dim=1)
        film_params = self.film_generator(film_input)
        gamma, beta = torch.split(film_params, self.h_dim, dim=-1)
        modulated_h = (1 + gamma) * canvas + beta
        h = self.decoder_backbone(modulated_h)
        px_rna_scale = self.px_rna_scale_decoder(h)
        if lib_size is not None:
            if lib_size.dim() == 1:
                lib_size = lib_size.unsqueeze(1)
            px_rna_scale = px_rna_scale * lib_size
        px_rna_rate = self.px_rna_rate_decoder(h)
        px_rna_dropout = self.px_rna_dropout_decoder(h)
        return dict(
            px_rna_scale=px_rna_scale,
            px_rna_rate=px_rna_rate,
            px_rna_dropout=px_rna_dropout,
            h=h,
        )

    def sample_batch(self, n_per_batch: int, **kwargs) -> Tuple[Any, ...]:
        """kwargs absorb ``replace`` for API compatibility with the legacy trainer (unused)."""
        _ = kwargs
        idxA = (np.arange(self._ptrA, self._ptrA + n_per_batch) % self.nA).astype(np.int64)
        self._ptrA = (self._ptrA + n_per_batch) % self.nA
        idxB = (np.arange(self._ptrB, self._ptrB + n_per_batch) % self.nB).astype(np.int64)
        self._ptrB = (self._ptrB + n_per_batch) % self.nB

        def dense_batch(matrix, idx):
            return torch.from_numpy(_to_dense_rowslice(matrix, idx)).float().to(self.device)

        X_A_counts = dense_batch(self.X_A_counts, idxA)
        X_B_counts = dense_batch(self.X_B_counts, idxB)
        X_A_log = dense_batch(self.X_A_log, idxA)
        X_B_log = dense_batch(self.X_B_log, idxB)
        libA = self.libA_global[idxA]
        libB = self.libB_global[idxB]
        coords_A = self.coords_A[idxA]
        coords_B = self.coords_B[idxB]
        return X_A_log, X_B_log, X_A_counts, X_B_counts, libA, libB, coords_A, coords_B

    def counterfactual_loss(
        self,
        XAA: torch.Tensor,
        XAB: torch.Tensor,
        XBB: torch.Tensor,
        XBA: torch.Tensor,
        XA_real: torch.Tensor,
        XB_real: torch.Tensor,
        libA: torch.Tensor,
        libB: torch.Tensor,
    ) -> torch.Tensor:
        ts = self.cfg.counterfactual_target_sum
        XAA_log = torch.log1p((XAA / (libA + 1e-8)) * ts)
        XAB_log = torch.log1p((XAB / (libA + 1e-8)) * ts)
        XBB_log = torch.log1p((XBB / (libB + 1e-8)) * ts)
        XBA_log = torch.log1p((XBA / (libB + 1e-8)) * ts)
        d_pred_AB = XAB_log - XAA_log
        d_pred_BA = XBA_log - XBB_log
        d_real_AB = XB_real - XA_real
        d_real_BA = XA_real - XB_real
        cos_ab = F.cosine_similarity(d_pred_AB, d_real_AB, dim=-1).mean()
        cos_ba = F.cosine_similarity(d_pred_BA, d_real_BA, dim=-1).mean()
        loss_pattern = 1 - 0.5 * (cos_ab + cos_ba)
        loss_mag = 0.5 * (F.mse_loss(d_pred_AB, d_real_AB) + F.mse_loss(d_pred_BA, d_real_BA))
        return self.cfg.cf_alpha * loss_pattern + self.cfg.cf_beta * loss_mag

    def forward(
        self,
        X_A_log: torch.Tensor,
        X_B_log: torch.Tensor,
        X_A_counts: torch.Tensor,
        X_B_counts: torch.Tensor,
        coords_A: torch.Tensor,
        coords_B: torch.Tensor,
        libA: torch.Tensor,
        libB: torch.Tensor,
    ) -> Tuple[Mapping, Mapping, Dict[str, torch.Tensor]]:
        qA = self._encode_batch(X_A_log, "A")
        qB = self._encode_batch(X_B_log, "B")
        joint_A = self._z_from_q(qA)
        joint_B = self._z_from_q(qB)
        z_A, z_B = joint_A["z"], joint_B["z"]
        mu_A, mu_B = joint_A["mu"], joint_B["mu"]
        var_A, var_B = joint_A["var"], joint_B["var"]
        n = z_A.shape[0]
        cond_A = torch.zeros(n, dtype=torch.long, device=self.device)
        cond_B = torch.ones(n, dtype=torch.long, device=self.device)

        outA = self._decode_z(z_A, cond_A, coords_A, libA)
        outB = self._decode_z(z_B, cond_B, coords_B, libB)
        outAB = self._decode_z(z_A, cond_B, coords_A, libA)
        outBA = self._decode_z(z_B, cond_A, coords_B, libB)
        XAA, XAB = outA["px_rna_scale"], outAB["px_rna_scale"]
        XBB, XBA = outB["px_rna_scale"], outBA["px_rna_scale"]

        cf = self.counterfactual_loss(XAA, XAB, XBB, XBA, X_A_log, X_B_log, libA, libB)
        recon_A = zinb_reconstruction_loss(
            X_A_counts,
            mu=outA["px_rna_scale"],
            theta=outA["px_rna_rate"].exp(),
            gate_logits=outA["px_rna_dropout"],
            reduction="mean",
        )
        recon_B = zinb_reconstruction_loss(
            X_B_counts,
            mu=outB["px_rna_scale"],
            theta=outB["px_rna_rate"].exp(),
            gate_logits=outB["px_rna_dropout"],
            reduction="mean",
        )

        fb = self.cfg.free_bits
        kld_A = torch.clamp(
            torch.distributions.kl_divergence(
                Normal(mu_A, var_A.sqrt()), Normal(torch.zeros_like(mu_A), torch.ones_like(var_A))
            ),
            min=fb,
        ).sum(dim=1)
        kld_B = torch.clamp(
            torch.distributions.kl_divergence(
                Normal(mu_B, var_B.sqrt()), Normal(torch.zeros_like(mu_B), torch.ones_like(var_B))
            ),
            min=fb,
        ).sum(dim=1)
        kld = 0.5 * (kld_A + kld_B)

        losses = {
            "recon_A": recon_A,
            "recon_B": recon_B,
            "kld": kld,
            "counterfactual": cf,
        }
        return joint_A, joint_B, losses

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _weights_step(self, epoch: int, tc: TrainingConfig) -> Tuple[float, float]:
        if epoch < tc.kl_warmup_epochs:
            kl_w = tc.kl_weight * epoch / max(tc.kl_warmup_epochs, 1)
        else:
            kl_w = tc.kl_weight
        cf_w = tc.cf_pre_weight if epoch < tc.cf_start_epoch else tc.counterfactual_weight
        return kl_w, cf_w

    def _weights_ramp(self, epoch: int, tc: TrainingConfig) -> float:
        if tc.n_epochs_kl_warmup > 0 and epoch <= tc.n_epochs_kl_warmup:
            p = epoch / tc.n_epochs_kl_warmup
            return tc.kl_weight * (1 - math.cos(p * math.pi / 2))
        return tc.kl_weight

    def fit(self, training: Optional[TrainingConfig] = None) -> Dict[str, List[float]]:
        """
        Unified training loop. Use ``TrainingConfig.schedule`` to select STEP (Fig. 3) vs RAMP (DLPFC sim).
        """
        tc = training or TrainingConfig()
        self._set_seed(tc.seed)
        self.to(self.device)

        opt = torch.optim.AdamW(self.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=tc.scheduler_factor,
            patience=tc.scheduler_patience,
        )

        history: Dict[str, List[float]] = {
            "epoch_recon_A": [],
            "epoch_recon_B": [],
            "epoch_kld": [],
            "epoch_counterfactual": [],
            "epoch_total": [],
        }

        n_samples = min(self.nA, self.nB)
        steps = (n_samples + tc.n_per_batch - 1) // tc.n_per_batch

        for epoch in range(1, tc.max_epoch + 1):
            self.train()
            if tc.schedule == TrainingSchedule.STEP:
                kl_w, cf_w = self._weights_step(epoch, tc)
            else:
                kl_w = self._weights_ramp(epoch, tc)
                cf_w = tc.counterfactual_weight

            sums = {
                "recon_A": 0.0,
                "recon_B": 0.0,
                "kld": 0.0,
                "counterfactual": 0.0,
                "total": 0.0,
            }
            n_batches = 0

            pbar = tqdm(range(steps), desc=f"Epoch {epoch}/{tc.max_epoch}", disable=not tc.verbose)
            for _ in pbar:
                pack = self.sample_batch(tc.n_per_batch, replace=tc.replace_sampling)
                X_A_log, X_B_log, X_A_c, X_B_c, libA, libB, cA, cB = pack

                _, _, losses = self.forward(
                    X_A_log,
                    X_B_log,
                    X_A_c,
                    X_B_c,
                    cA,
                    cB,
                    libA,
                    libB,
                )

                ra = losses["recon_A"].mean()
                rb = losses["recon_B"].mean()
                kld = losses["kld"].mean()
                cf = losses["counterfactual"].mean()
                if tc.counterfactual_clamp_max is not None:
                    cf = torch.clamp(cf, max=tc.counterfactual_clamp_max)

                total = tc.recon_w * (ra + rb) + kl_w * kld + cf_w * cf

                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), tc.grad_clip_norm)
                opt.step()

                bs = X_A_log.size(0)
                sums["recon_A"] += ra.item() * bs
                sums["recon_B"] += rb.item() * bs
                sums["kld"] += kld.item() * bs
                sums["counterfactual"] += cf.item() * bs
                sums["total"] += total.item() * bs
                n_batches += bs

                pbar.set_postfix(
                    total=f"{sums['total'] / n_batches:.3f}",
                    recon_A=f"{ra:.3f}",
                    recon_B=f"{rb:.3f}",
                    kl_w=f"{kl_w:.3f}",
                    kld=f"{kld:.3f}",
                    cf_w=f"{cf_w:.3g}",
                    counterfactual=f"{cf:.3f}",
                )

            avg_total = sums["total"] / n_batches
            sched.step(avg_total)
            history["epoch_recon_A"].append(sums["recon_A"] / n_batches)
            history["epoch_recon_B"].append(sums["recon_B"] / n_batches)
            history["epoch_kld"].append(sums["kld"] / n_batches)
            history["epoch_counterfactual"].append(sums["counterfactual"] / n_batches)
            history["epoch_total"].append(avg_total)

        self.trained_state_dict = deepcopy(self.state_dict())
        return history

    @torch.no_grad()
    def get_all_predictions(
        self,
        adata: ad.AnnData,
        *,
        which_enc: str = "A",
        batch_size: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        device = next(self.parameters()).device
        x_data = adata.layers["log_norm"]
        coords = adata.obsm["spatial"]
        if "library_size" in adata.obs:
            lib_size = np.asarray(adata.obs["library_size"]).ravel()
        else:
            c = adata.layers["count"]
            lib_size = np.asarray(c.sum(axis=1)).ravel()

        reals, cfs = [], []
        for i in range(0, x_data.shape[0], batch_size):
            stop = min(i + batch_size, x_data.shape[0])
            sub_x = _to_dense_rowslice(x_data, np.arange(i, stop))
            sub_coords = np.asarray(coords[i:stop], dtype=np.float32)
            xt = torch.as_tensor(sub_x, dtype=torch.float32, device=device)
            ct = torch.as_tensor(sub_coords, dtype=torch.float32, device=device)
            lt = torch.as_tensor(lib_size[i:stop], dtype=torch.float32, device=device).unsqueeze(1)

            enc = self.encoder_A if which_enc == "A" else self.encoder_B
            z = self._z_from_q(enc(xt))["mu"]
            n = z.shape[0]
            c0 = torch.zeros(n, dtype=torch.long, device=device)
            c1 = torch.ones(n, dtype=torch.long, device=device)
            r_A = self._decode_z(z, c0, ct, lt)["px_rna_scale"]
            r_B = self._decode_z(z, c1, ct, lt)["px_rna_scale"]
            reals.append(r_A.cpu().numpy())
            cfs.append(r_B.cpu().numpy())

        return np.concatenate(reals, axis=0), np.concatenate(cfs, axis=0)

    def compute_drift_score(
        self,
        adata_A: ad.AnnData,
        adata_B: ad.AnnData,
        **kwargs: Any,
    ):
        """See :func:`counterfactual_spatial_cvae.drift_score.compute_drift_score_v6`."""
        from .drift_score import compute_drift_score as _fn

        return _fn(adata_A, adata_B, self, **kwargs)
