#!/usr/bin/env python3
"""
Visual sanity-check after each training stage.
Saves a PNG grid to <out_dir>/samples_<stage>.png.

Usage (called automatically by run_pipeline.sh):
  python scripts/sample_vis.py fdp --ckpt <path> --out_dir <dir>
  python scripts/sample_vis.py vae --ckpt <path> [--config <cfg.yaml>] --out_dir <dir>
  python scripts/sample_vis.py ldm --ckpt <path> [--config <cfg.yaml>] --out_dir <dir>

If --config is omitted for vae/ldm, the script looks for
  <ckpt_parent_dir>/configs/<stage>_pipeline.yaml
(i.e. the configs/ sibling of the checkpoints/ directory).
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_slices(data_root: str, n: int, seed=42, size: int = 256):
    """Return n middle slices from random NIfTI files as (H,W) float32 in [-1,1]."""
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    all_files = sorted(glob.glob(os.path.join(data_root, "**", "*.nii*"), recursive=True))
    # Prefer T2 files (BraTS naming: *_t2.nii); fall back to all NIfTI files (IXI flat layout)
    files = [f for f in all_files if os.path.basename(f).lower().endswith(("_t2.nii", "_t2.nii.gz"))]
    if not files:
        files = all_files
    if not files:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in {data_root}")
    chosen = rng.choice(files, size=min(n, len(files)), replace=False)
    slices = []
    for path in chosen:
        vol = nib.load(path).get_fdata().astype(np.float32)
        mid = vol.shape[-1] // 2
        sl = vol[..., mid]
        vmax = sl.max()
        if vmax <= 0:
            continue
        sl = sl / vmax * 2.0 - 1.0          # [-1, 1]
        if sl.shape[0] != size or sl.shape[1] != size:
            t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            sl = t.squeeze().numpy()
        slices.append(sl)
    return slices


def _t(arr: np.ndarray) -> torch.Tensor:
    """(H,W) numpy → (1,1,H,W) float tensor."""
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


def _np(t: torch.Tensor) -> np.ndarray:
    """Tensor → (H,W) numpy, clamped to [-1,1]."""
    return t.squeeze().detach().cpu().float().clamp(-1, 1).numpy()


def _thresh_diff(diff: np.ndarray, percentile: float = 97.0,
                 min_cc_area: int = 100, x: np.ndarray = None) -> np.ndarray:
    """Paper post-processing for a single 2D slice.
    Steps: mean filter (size=5) → brain-masked percentile threshold → remove small CCs."""
    from scipy.ndimage import uniform_filter
    from skimage.measure import label, regionprops
    smoothed = uniform_filter(np.abs(diff), size=5)
    brain_mask = (x > -0.99) if x is not None else np.ones_like(smoothed, dtype=bool)
    brain_vals = smoothed[brain_mask]
    thr = np.percentile(brain_vals, percentile) if brain_vals.size > 0 else 0.0
    binary = (smoothed > thr) & brain_mask
    cc = label(binary, connectivity=2)
    for prop in regionprops(cc):
        if prop.filled_area <= min_cc_area:  # default 50
            binary[cc == prop.label] = False
    return binary.astype(np.float32)


def _save_grid(rows: list, col_titles: list, out_path: str):
    """rows: list of lists of (H,W) arrays; col_titles: column header strings."""
    nrows, ncols = len(rows), len(rows[0])
    _, axes = plt.subplots(nrows, ncols,
                           figsize=(ncols * 3.2, nrows * 3.2),
                           squeeze=False)
    for r, row_imgs in enumerate(rows):
        for c, img in enumerate(row_imgs):
            ax = axes[r][c]
            if "diff" in col_titles[c].lower() or "|" in col_titles[c]:
                ax.imshow(np.abs(img), cmap="hot", vmin=0, vmax=1)
            else:
                ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, fontweight="bold")
    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def _find_fdp_ckpt(vae_ckpt: str) -> str:
    """
    Given a VAE checkpoint path, locate the FDP checkpoint in the sibling fdp/
    directory of the same run.

    Looks for (in order): best.ckpt → last.ckpt inside <run_root>/fdp/checkpoints/.
    Raises FileNotFoundError if not found.
    """
    # vae_ckpt is something like  <run_root>/vae/checkpoints/last.ckpt
    # Walk up until we find a directory that contains both "vae" and "fdp" children.
    path = os.path.realpath(vae_ckpt)
    for _ in range(5):
        path = os.path.dirname(path)
        fdp_ckpt_dir = os.path.join(path, "fdp", "checkpoints")
        if os.path.isdir(fdp_ckpt_dir):
            for name in ("best.ckpt", "last.ckpt"):
                candidate = os.path.join(fdp_ckpt_dir, name)
                if os.path.exists(candidate):
                    return candidate
    raise FileNotFoundError(
        f"Could not find FDP checkpoint relative to VAE ckpt: {vae_ckpt}\n"
        f"Expected at <run_root>/fdp/checkpoints/best.ckpt or last.ckpt"
    )


def _best_ckpt(ckpt_dir: str) -> str:
    """
    Return best checkpoint path.
    For FDP: best.ckpt symlink if it exists, else last.ckpt.
    For Lightning (VAE/LDM): read best_model_path from last.ckpt's callback state.
    """
    best_link = os.path.join(ckpt_dir, "best.ckpt")
    if os.path.exists(best_link):
        return best_link

    last = os.path.join(ckpt_dir, "last.ckpt")
    if not os.path.exists(last):
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    try:
        sd = torch.load(last, map_location="cpu", weights_only=False)
        for cb_state in sd.get("callbacks", {}).values():
            if isinstance(cb_state, dict):
                best = cb_state.get("best_model_path", "")
                if best and os.path.exists(best):
                    return best
    except Exception:
        pass
    return last


# ---------------------------------------------------------------------------
# FDP sampling
# ---------------------------------------------------------------------------

def sample_fdp(ckpt: str, out_dir: str, data_root: str, n: int, device: str):
    from fdp.module import FDPModule

    print(f"[FDP] ckpt: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("fdp_config") or payload.get("config") or {}
    from fdp.module import _extract_fdp_state
    model = FDPModule(
        n_prior=cfg.get("n_prior", 128),
        mask_ratio=cfg.get("mask_ratio", 0.10),
        img_size=cfg.get("img_size", 256),
        prior_path_real=os.path.join(ROOT, "preload/t2_hf_0d3_embed_128ac_real_frq.pth"),
        prior_path_imag=os.path.join(ROOT, "preload/t2_hf_0d3_embed_128ac_imag_frq.pth"),
    )
    model.load_state_dict(_extract_fdp_state(payload))
    model.eval().to(device)

    slices = _load_slices(data_root, n)
    rows = []
    for sl in slices:
        x = _t(sl).to(device)
        with torch.no_grad():
            frec, hf_supp = model(x)
        rows.append([_np(x), _np(frec), _np(hf_supp)])

    _save_grid(rows, ["Input", "FDP-frec", "HF-supp"],
               os.path.join(out_dir, "samples_fdp.png"))


# ---------------------------------------------------------------------------
# VAE sampling
# ---------------------------------------------------------------------------

def sample_vae(ckpt: str, config: str, out_dir: str, data_root: str,
               n: int, device: str):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from fdp.module import FDPModule

    print(f"[VAE] ckpt: {ckpt}")
    cfg = OmegaConf.load(config)
    model = instantiate_from_config(cfg.model)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd), strict=False)
    model.eval().to(device)

    fdp_ckpt = _find_fdp_ckpt(ckpt)
    print(f"[VAE] FDP ckpt: {fdp_ckpt}")
    payload = torch.load(fdp_ckpt, map_location="cpu", weights_only=False)
    cfg_fdp = payload.get("fdp_config") or payload.get("config") or {}
    fdp_model = FDPModule(
        n_prior=cfg_fdp.get("n_prior", 128),
        mask_ratio=cfg_fdp.get("mask_ratio", 0.10),
        img_size=cfg_fdp.get("img_size", 256),
        prior_path_real=os.path.join(ROOT, "preload/t2_hf_0d3_embed_128ac_real_frq.pth"),
        prior_path_imag=os.path.join(ROOT, "preload/t2_hf_0d3_embed_128ac_imag_frq.pth"),
    )
    from fdp.module import _extract_fdp_state
    fdp_model.load_state_dict(_extract_fdp_state(payload))
    fdp_model.eval().to(device)

    slices = _load_slices(data_root, n)
    rows = []
    for sl in slices:
        x = _t(sl).to(device)
        with torch.no_grad():
            rec_x = model.decode(model.encode(x).sample())
            frec, hf = fdp_model(x)
            rec_frec = model.decode(model.encode(frec).sample())
            rec_hf   = model.decode(model.encode(hf).sample())
            row = [_np(x), _np(rec_x), _np(x - rec_x),
                   _np(frec), _np(rec_frec), _np(hf), _np(rec_hf)]
        rows.append(row)

    _save_grid(rows,
               ["Input", "VAE(Input)", "|Diff|",
                "FDP-frec", "VAE(FDP-frec)", "FDP-hf_supp", "VAE(FDP-hf_supp)"],
               os.path.join(out_dir, "samples_vae.png"))


# ---------------------------------------------------------------------------
# LDM sampling
# ---------------------------------------------------------------------------

def sample_ldm(ckpt: str, config: str, out_dir: str, data_root: str,
               n: int, device: str):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    print(f"[LDM] ckpt: {ckpt}")
    cfg = OmegaConf.load(config)
    cfg.model.params.ckpt_path = ckpt
    model = instantiate_from_config(cfg.model)
    model.eval().to(device)

    slices = _load_slices(data_root, n)
    rows = []
    for sl in slices:
        x = _t(sl).to(device)
        with torch.no_grad():
            frec, hf_supp = model.cond_stage_model(x)
            c = model.get_learned_conditioning(x)
            c_frec, c_hf = c.chunk(2, dim=1)
            frec_dec = model.decode_first_stage(c_frec)
            hf_dec   = model.decode_first_stage(c_hf)
            samples, _ = model.sample_log(
                cond=c, batch_size=1, ddim=True,
                ddim_steps=50, eta=0.0,
                x0=None, mask=None, noise_dropout=0.0,
            )
            rec_raw = model.decode_first_stage(samples)
            rec = model.cond_stage_model.enhance(x, rec_raw)
        x_np = _np(x)
        diff = _np(x - rec)
        rows.append([x_np, _np(frec), _np(frec_dec), _np(hf_supp), _np(hf_dec), _np(rec), diff])

    _save_grid(rows, ["Input", "FDP-frec", "VAE(frec)", "FDP-hf", "VAE(hf)", "LDM-rec", "|Diff|"],
               os.path.join(out_dir, "samples_ldm.png"))


# ---------------------------------------------------------------------------
# Config auto-discovery
# ---------------------------------------------------------------------------

def _find_config(ckpt: str, stage: str) -> str:
    """Return <stage>_pipeline.yaml from the configs/ sibling of checkpoints/."""
    ckpt_dir  = os.path.dirname(os.path.realpath(ckpt))
    stage_dir = os.path.dirname(ckpt_dir)
    candidate = os.path.join(stage_dir, "configs", f"{stage}_pipeline.yaml")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"--config not given and auto-discovery failed.\n"
        f"Expected: {candidate}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fdp", "vae", "ldm"])
    p.add_argument("--ckpt",      required=True, help="path to checkpoint file or dir")
    p.add_argument("--config",    default=None,  help="YAML config (VAE/LDM only)")
    p.add_argument("--out_dir",   required=True)
    p.add_argument("--data_root", default=os.path.join(ROOT, "data/IXI-T2-hd"))
    p.add_argument("--n",         type=int, default=6)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # If a directory is passed, find the best checkpoint inside it
    ckpt = _best_ckpt(args.ckpt) if os.path.isdir(args.ckpt) else args.ckpt
    print(f"Using checkpoint: {ckpt}")

    kw = dict(ckpt=ckpt, out_dir=args.out_dir,
              data_root=args.data_root, n=args.n, device=args.device)

    if args.stage == "fdp":
        sample_fdp(**kw)
    elif args.stage == "vae":
        config = args.config or _find_config(ckpt, "vae")
        sample_vae(config=config, **kw)
    else:
        config = args.config or _find_config(ckpt, "ldm")
        sample_ldm(config=config, **kw)


if __name__ == "__main__":
    main()
