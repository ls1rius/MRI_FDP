"""
Initialise the FRM prior context bank with k-means++ cluster centres.

Clusters the full low-frequency 76×76 (int(256*0.3)) patch for both real and
imaginary components separately, saving [n_prior, H_lf*H_lf] tensors.

Usage (from project root):
    python fdp/init_prior.py \\
        --data_root data/IXI-T2-hd \\
        --output_dir preload \\
        --n_prior 128
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdp.data import MRIDataset
from fdp.ops import _fft_shift, _to_unit


def _collect_patches(data_root, img_size, H_lf, max_slices):
    """Extract the full H_lf×H_lf centred low-freq FFT patch from every slice."""
    dataset = MRIDataset(data_root=data_root, size=img_size, is_test=True)
    crow = ccol = img_size // 2
    h2 = H_lf // 2
    reals, imags = [], []
    for item in tqdm(dataset, desc="extracting patches"):
        vol = torch.from_numpy(item["image"]).float()
        _, _, D = vol.shape
        for s in range(D):
            slc = vol[:, :, s: s + 1].permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                f = _fft_shift(_to_unit(slc))
            patch = f[:, :, crow - h2: crow + h2, ccol - h2: ccol + h2]
            reals.append(patch.real.flatten().numpy())
            imags.append(patch.imag.flatten().numpy())
            if len(reals) >= max_slices:
                break
        if len(reals) >= max_slices:
            break
    return np.stack(reals, axis=0), np.stack(imags, axis=0)


def _kmeans(patches, n_clusters, minibatch=False):
    """Fit KMeans (or MiniBatchKMeans) and return cluster centres as float32 array."""
    if minibatch:
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=0,
            n_init=3,
            batch_size=4096,
        )
    else:
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=0,
            n_init=32,
            max_iter=1280,
        )
    km.fit(patches)
    return km.cluster_centers_.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_prior",    type=int,   default=128)
    parser.add_argument("--img_size",   type=int,   default=256)
    parser.add_argument("--lf_ratio",   type=float, default=0.3,
                        help="prior stored at this fraction of img_size (default 0.3)")
    parser.add_argument("--max_slices", type=int,   default=50_000)
    parser.add_argument("--minibatch",  action="store_true",
                        help="use MiniBatchKMeans instead of KMeans (faster but less accurate)")
    args = parser.parse_args()

    H_lf = int(args.img_size * args.lf_ratio)
    print(f"Prior size: {H_lf}×{H_lf} = {H_lf*H_lf} dims")

    print("Collecting FFT patches …")
    reals, imags = _collect_patches(args.data_root, args.img_size, H_lf, args.max_slices)
    print(f"  collected {len(reals)} slices  patch shape: {reals.shape}")

    algo = "MiniBatchKMeans" if args.minibatch else "KMeans"
    print(f"{algo} (k={args.n_prior}) on real component …")
    centres_real = _kmeans(reals, args.n_prior, minibatch=args.minibatch)

    print(f"{algo} (k={args.n_prior}) on imaginary component …")
    centres_imag = _kmeans(imags, args.n_prior, minibatch=args.minibatch)

    os.makedirs(args.output_dir, exist_ok=True)
    name = f"t2_hf_0d{int(args.lf_ratio*10)}_embed_{args.n_prior}ac"
    out_r = os.path.join(args.output_dir, f"{name}_real_frq.pth")
    out_i = os.path.join(args.output_dir, f"{name}_imag_frq.pth")
    torch.save(torch.from_numpy(centres_real), out_r)
    torch.save(torch.from_numpy(centres_imag), out_i)
    print(f"Saved:\n  {out_r}  {centres_real.shape}\n  {out_i}  {centres_imag.shape}")


if __name__ == "__main__":
    main()
