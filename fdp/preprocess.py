"""
Batch FDP preprocessing: apply a trained FDPModule to an MRI dataset and
save conditioning signals (frec, hf_supp) for downstream Stage-2 LDM training.

Output layout:
    <output_dir>/
        frec/<subject_id>/<slice_idx>.pt
        hf_supp/<subject_id>/<slice_idx>.pt

Usage (from project root):
    python fdp/preprocess.py \\
        --ckpt outputs/fdp_latest.pt \\
        --data_root ./data/BraTS/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \\
        --output_dir data/BraTS/fdp_cond
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdp.module import FDPModule
from fdp.data import MRIDataset


@torch.no_grad()
def preprocess(model, data_root, output_dir, img_size, device, batch_size=8):
    dataset = MRIDataset(data_root=data_root, size=img_size, is_test=True)

    for item in tqdm(dataset, desc="volumes"):
        vol = torch.from_numpy(item["image"]).float()
        path = item["file_path"]
        subject_id = os.path.splitext(os.path.basename(path))[0].replace(".nii", "")

        frec_dir  = os.path.join(output_dir, "frec",    subject_id)
        hfsup_dir = os.path.join(output_dir, "hf_supp", subject_id)
        os.makedirs(frec_dir,  exist_ok=True)
        os.makedirs(hfsup_dir, exist_ok=True)

        H, W, D = vol.shape
        for start in range(0, D, batch_size):
            slices = vol[:, :, start: start + batch_size]
            x = slices.permute(2, 0, 1).unsqueeze(1).to(device)
            frec, hf_supp = model(x)
            for i, s_idx in enumerate(range(start, start + x.shape[0])):
                torch.save(frec[i].cpu(),    os.path.join(frec_dir,  f"{s_idx}.pt"))
                torch.save(hf_supp[i].cpu(), os.path.join(hfsup_dir, f"{s_idx}.pt"))

    print(f"Done. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size",   type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = FDPModule.from_checkpoint(args.ckpt).to(device).eval()
    preprocess(
        model=model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        img_size=args.img_size,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
