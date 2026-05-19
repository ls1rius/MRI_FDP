"""
FDPModule: the Frequency-Decomposition Preprocessing module.

Reference:
  "FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised
  Anomaly Detection in Brain MRI", AAAI 2026.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _extract_fdp_state(ckpt: dict) -> dict:
    """Return FDPModule state_dict from either old or Lightning checkpoint format."""
    if "fdp_model" in ckpt:          # new Lightning format
        return ckpt["fdp_model"]
    if "model" in ckpt:              # old custom-loop format
        return ckpt["model"]
    if "state_dict" in ckpt:         # raw Lightning state_dict (model. prefix)
        sd = ckpt["state_dict"]
        return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    raise KeyError(f"Cannot extract FDP state dict from checkpoint keys: {list(ckpt.keys())}")

from .ops import (
    mri_norm,
    get_low_frequency,
    combine_low_high,
    remove_low_frequency,
    enhance_frequency,
)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention used inside the FRM.
    Matches the x-transformer Attention(dim, dim_head=64, heads=8) in the original code.
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner = dim_head * heads

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        N, _, dim = q.shape
        h = self.heads
        Q = self.to_q(q).reshape(N, 1,  h, self.dim_head).transpose(1, 2)
        K = self.to_k(kv).reshape(N, -1, h, self.dim_head).transpose(1, 2)
        V = self.to_v(kv).reshape(N, -1, h, self.dim_head).transpose(1, 2)
        attn = self.attn_drop(F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(N, 1, h * self.dim_head)
        return self.to_out(out)


class _FRMBlock(nn.Module):
    """One FRM block (real or imaginary): BN → cross-attention → conv refinement."""

    def __init__(self, attn_dim: int, n_prior: int, H_lf: int, dim_head: int = 64, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_prior = n_prior
        self.H_lf = H_lf
        self.attn_dim = attn_dim

        self.bn_in = nn.BatchNorm2d(1)
        self.attn = CrossAttention(attn_dim, dim_head=dim_head, heads=heads, dropout=dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_bn = self.bn_in(x)
        prior_bn = self.bn_in(prior.reshape(N * self.n_prior, 1, H, W))
        prior_bn = prior_bn.reshape(N, self.n_prior, H * W)
        out = self.attn(x_bn.flatten(1).unsqueeze(1), prior_bn)
        out = out.reshape(N, C, H, W)
        return self.conv(out)


class FDPModule(nn.Module):
    """
    Frequency-Decomposition Preprocessing (FDP) module.

    Produces two conditioning signals for downstream generative models:
      frec    — frequency-reconstructed "healthy" MRI
      hf_supp — high-frequency supplement of the original input

    Args:
        n_prior:          number of learnable prior context prototypes (default 128)
        mask_ratio:       low-frequency region fraction (paper default 0.10)
        img_size:         spatial size of input, assumed square (default 256)
        prior_path_real:  path to pre-initialised real prior .pth
        prior_path_imag:  path to pre-initialised imaginary prior .pth
        ckpt_path:        path to a full Stage-1 checkpoint; loads the entire state dict
        dim_head:         per-head dimension in cross-attention (default 64)
        heads:            number of attention heads (default 8)
    """

    _LF_RATIO = 0.3  # prior stored at 30 % of img_size

    def __init__(
        self,
        n_prior: int = 128,
        mask_ratio: float = 0.10,
        img_size: int = 256,
        prior_path_real: Optional[str] = None,
        prior_path_imag: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.n_prior = n_prior

        H_lf = int(img_size * self._LF_RATIO)
        self.H_lf = H_lf
        mh = int(img_size * mask_ratio // 2 * 2)
        self.mh = mh
        attn_dim = mh * mh

        def _load(path, name):
            if not path:
                # No path provided — return placeholder zeros; the real values will
                # be loaded from the checkpoint state dict (e.g. via from_checkpoint).
                return torch.zeros(n_prior * H_lf * H_lf)
            if not os.path.exists(path):
                raise FileNotFoundError(f"FDPModule: {name} not found at '{path}'")
            return torch.load(path, map_location="cpu", weights_only=True).reshape(-1)

        self.prior_real = nn.Parameter(_load(prior_path_real, "prior_path_real"))
        self.prior_imag = nn.Parameter(_load(prior_path_imag, "prior_path_imag"))

        self.frm_real = _FRMBlock(attn_dim, n_prior, H_lf, dim_head, heads, dropout)
        self.frm_imag = _FRMBlock(attn_dim, n_prior, H_lf, dim_head, heads, dropout)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.load_state_dict(_extract_fdp_state(ckpt))

    def _crop_prior(self, prior_param: nn.Parameter, N: int) -> torch.Tensor:
        proto = prior_param.reshape(self.n_prior, self.H_lf, self.H_lf)
        proto = proto.unsqueeze(0).expand(N, -1, -1, -1)
        crow = ccol = self.H_lf // 2
        h2 = self.mh // 2
        return proto[:, :, crow - h2: crow + h2, ccol - h2: ccol + h2].contiguous()

    def _predict_low_freq(self, fl: torch.Tensor) -> torch.Tensor:
        N = fl.shape[0]
        prior_r = self._crop_prior(self.prior_real, N)
        # prior_i = self._crop_prior(self.prior_imag, N)

        pred_r = self.frm_real(fl.real, prior_r)
        # Imag branch is not trained: near DC the imaginary component is theoretically ~0
        # so zeroing it has negligible effect on the reconstructed magnitude image.
        # To re-enable imag training: uncomment the line below and remove zeros_like.
        # pred_i = self.frm_imag(fl.imag, prior_i)
        pred_i = torch.zeros_like(pred_r)
        # pred_i = torch.randn_like(pred_r) * 1e-6  # small random noise to break ties in magnitude when imag is zero
        return torch.complex(pred_r, pred_i)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, 1, H, W) MRI slice in [-1, 1]
        Returns:
            frec    : (N, 1, H, W) frequency-reconstructed healthy MRI
            hf_supp : (N, 1, H, W) high-frequency supplement
        """
        fl = get_low_frequency(x, self.mask_ratio)
        fl_pred = self._predict_low_freq(fl)
        frec    = combine_low_high(x, fl_pred, self.mask_ratio)
        hf_supp = remove_low_frequency(x, self.mask_ratio)
        return frec, hf_supp

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Stage-1 training losses for the FRM.

        Returns:
            loss       : scalar total loss for .backward()
            loss_dict  : {'loss', 'loss_real', 'loss_imag'}
        """
        fl = get_low_frequency(x, self.mask_ratio)
        N = x.shape[0]

        prior_r = self._crop_prior(self.prior_real, N)
        prior_i = self._crop_prior(self.prior_imag, N)

        pred_r = self.frm_real(fl.real, prior_r)
        pred_i = self.frm_imag(fl.imag, prior_i)

        loss_real = F.l1_loss(pred_r, fl.real)
        loss_imag = F.l1_loss(pred_i, fl.imag)
        loss = loss_real + loss_imag

        loss_dict = {
            "loss":      loss,
            "loss_real": loss_real,
            "loss_imag": loss_imag,
        }
        return loss, loss_dict

    def enhance(self, x_ref: torch.Tensor, y_rec: torch.Tensor) -> torch.Tensor:
        """Amplitude-alignment post-processing after generative reconstruction."""
        return enhance_frequency(x_ref, y_rec, self.mask_ratio)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, **kwargs) -> "FDPModule":
        """Load a trained FDPModule from a checkpoint saved by fdp/train.py."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("fdp_config") or ckpt.get("config") or {}
        model = cls(**{**cfg, **kwargs})
        model.load_state_dict(_extract_fdp_state(ckpt))
        return model
