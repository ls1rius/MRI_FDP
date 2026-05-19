"""
Pure frequency-domain operations for MRI preprocessing.

All functions take (N, C, H, W) float tensors in [-1, 1] and return the same.
No learnable parameters — safe to use in data pipelines or as utility functions.
"""
import torch
import torch.fft as fft


def mri_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalisation to [-1, 1]."""
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - mn) / (mx - mn + 1e-8) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_unit(x: torch.Tensor) -> torch.Tensor:
    """Normalise to [0, 1] (internal use before FFT)."""
    return (mri_norm(x) + 1.0) / 2.0


def _fft_shift(x: torch.Tensor) -> torch.Tensor:
    """2D FFT then fftshift so that DC (low-freq) is at the spatial centre."""
    return fft.fftshift(fft.fftn(x, dim=(-2, -1)), dim=(-2, -1))


def _ifft_shift(f: torch.Tensor) -> torch.Tensor:
    """Inverse of _fft_shift."""
    return fft.ifftn(fft.ifftshift(f, dim=(-2, -1)), dim=(-2, -1))


def _lf_roi(H: int, W: int, mask_ratio: float):
    """
    Return (r0, r1, c0, c1) slice indices for the centred low-frequency region.

    The region size is computed as int(dim * mask_ratio // 2 * 2) to ensure an
    even number of rows/cols, matching the original implementation.
    """
    mh = int(H * mask_ratio // 2 * 2)
    mw = int(W * mask_ratio // 2 * 2)
    r0, c0 = H // 2 - mh // 2, W // 2 - mw // 2
    return r0, r0 + mh, c0, c0 + mw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_low_frequency(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Extract the low-frequency patch from the centred FFT spectrum.

    Args:
        x:           (N, C, H, W) MRI tensor in [-1, 1]
        mask_ratio:  fraction of image size used for the low-freq region
    Returns:
        Complex tensor of shape (N, C, mH, mW)
    """
    f = _fft_shift(_to_unit(x))
    r0, r1, c0, c1 = _lf_roi(*x.shape[-2:], mask_ratio)
    return f[:, :, r0:r1, c0:c1].clone()


def combine_low_high(
    x: torch.Tensor, low_freq: torch.Tensor, mask_ratio: float
) -> torch.Tensor:
    """
    Replace the low-frequency centre of x's spectrum with *low_freq*, then IDFT.

    This is the MERGE step (eq. 11 in the paper):
        f̂ = MERGE(f̂_l, f_h)  →  Î = IDFT(f̂)

    Args:
        x:         (N, C, H, W) original MRI — supplies high-freq context
        low_freq:  (N, C, mH, mW) complex replacement for the low-freq centre
        mask_ratio: fraction of image size for the low-freq region
    Returns:
        (N, C, H, W) frequency-reconstructed MRI in [-1, 1]
    """
    x_unit = _to_unit(x)
    f = _fft_shift(x_unit)
    r0, r1, c0, c1 = _lf_roi(*x.shape[-2:], mask_ratio)
    f[:, :, r0:r1, c0:c1] = low_freq
    out = torch.abs(_ifft_shift(f))
    # Preserve the skull-stripped background mask from the input
    out = torch.where(x_unit > 0, out, torch.zeros_like(out))
    return mri_norm(out)


def remove_low_frequency(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    High-pass filter: zero out the low-freq centre and IDFT back.

    Produces the high-frequency supplement (HFSup, eq. 9 in the paper):
        f_h = f ⊙ H_hp  →  I_h = IDFT(f_h)

    Args:
        x:           (N, C, H, W) MRI tensor in [-1, 1]
        mask_ratio:  fraction of image size for the low-freq region to remove
    Returns:
        (N, C, H, W) high-frequency image in [-1, 1]
    """
    x_unit = _to_unit(x)
    f = _fft_shift(x_unit)
    r0, r1, c0, c1 = _lf_roi(*x.shape[-2:], mask_ratio)
    f[:, :, r0:r1, c0:c1] = 0
    out = torch.abs(_ifft_shift(f))
    out = torch.where(x_unit > 0, out, torch.zeros_like(out))
    return mri_norm(out)


def enhance_frequency(
    x_ref: torch.Tensor, y_rec: torch.Tensor, mask_ratio: float,
    energy_scale: bool = True,
) -> torch.Tensor:
    """
    Post-processing: apply background mask and normalise *y_rec*.

    When energy_scale=True, independently scale the low-freq and high-freq
    complex spectrum of *y_rec* so each band's L2 energy matches *x_ref*
    before reconstructing the image.

    Args:
        x_ref:        (N, C, H, W) original MRI (reference)
        y_rec:        (N, C, H, W) generative-model reconstruction
        mask_ratio:   low-frequency boundary
        energy_scale: if True, equalise per-band energy to x_ref (default: True)
    Returns:
        (N, C, H, W) enhanced reconstruction in [-1, 1],
        with background pixels reset to -1.
    """
    if energy_scale:
        x_unit = _to_unit(x_ref)
        y_unit = _to_unit(y_rec)
        N, C, H, W = x_unit.shape
        r0, r1, c0, c1 = _lf_roi(H, W, mask_ratio)

        fx = _fft_shift(x_unit)
        fy = _fft_shift(y_unit)

        mask_low = torch.zeros(N, C, H, W, dtype=torch.bool, device=x_ref.device)
        mask_low[:, :, r0:r1, c0:c1] = True
        mask_high = ~mask_low

        def _band_norm(f, m):
            flat_f = torch.abs(f).reshape(N, -1)
            flat_m = m.reshape(N, -1)
            return torch.stack([flat_f[i][flat_m[i]].norm() for i in range(N)])

        low_scale  = (_band_norm(fx, mask_low)  / (_band_norm(fy, mask_low)  + 1e-8)).view(N, 1, 1, 1)
        high_scale = (_band_norm(fx, mask_high) / (_band_norm(fy, mask_high) + 1e-8)).view(N, 1, 1, 1)
        scale = high_scale.expand(N, C, H, W).clone()
        scale[:, :, r0:r1, c0:c1] = low_scale
        out = torch.abs(_ifft_shift(fy * scale))
        out = mri_norm(out)
    else:
        out = mri_norm(_to_unit(y_rec))

    out = torch.where(x_ref > -1.0, out, torch.full_like(out, -1.0))
    return out.clamp(-1.0, 1.0)
