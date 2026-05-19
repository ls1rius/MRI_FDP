# FDP — Frequency-Decomposition Preprocessing for Unsupervised Anomaly Detection in Brain MRI

> **FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI**
> AAAI 2026

![](./assets/overview.png)

FDP is a plug-and-play preprocessing module that suppresses lesion-related low-frequency content in MRI by replacing it with a healthy reconstruction sampled from a learned prior context bank, then outputs two conditioning signals for a downstream generative model.

---

## Key Insight

MRI lesions (tumours, MS plaques, …) appear as smooth, homogeneous regions → they live primarily in **low-frequency** Fourier components.
Healthy brain anatomy shares **consistent** low-frequency patterns across subjects.
FDP exploits both: remove the low-freq lesion signal, reconstruct it from a healthy prior, and feed the result into the generative model.

---

## Project Structure

```
FDP/
├── fdp/                        # FDP module (standalone, no LDM dependency)
│   ├── ops.py                  # Pure FFT operations (no nn.Module)
│   ├── module.py               # FDPModule — FRM attention + full pipeline
│   ├── train.py                # Stage-1 training (FRM only)
│   ├── init_prior.py           # k-means++ prior initialisation (optional)
│   ├── preprocess.py           # Batch preprocessing → save frec / hf_supp
│   └── data/
│       └── dataset.py          # MRIDataset (NIfTI loader, recursive scan)
│
├── ldm/                        # Pure LDM (no frequency logic)
│   ├── models/
│   │   ├── autoencoder.py      # KL-VAE
│   │   ├── mri_vae.py          # MRI-specific VAE subclass
│   │   └── diffusion/
│   │       ├── ddpm.py         # Base LatentDiffusion
│   │       └── mri_ldm.py      # MRILatentDiffusion — uses FDPModule as cond_stage
│   ├── modules/                # UNet, attention, EMA, …
│   └── data/
│       └── mri.py              # MRI data loaders for LDM training
│
├── configs/
│   ├── fdp/                    # Stage-1 FDP training config
│   ├── vae/                    # Stage-2 VAE training config
│   ├── ldm/                    # Stage-3 LDM training config
│   └── eval/                   # Inference config
│
├── preload/                    # Pre-trained FRM prior context banks
│   ├── t2_hf_0d3_embed_128ac_real_frq.pth
│   └── t2_hf_0d3_embed_128ac_imag_frq.pth
│
├── scripts/
│   └── sample_vis.py           # Visual sampling / quality check tool
│
├── data/                       # Datasets (local)
│   ├── IXI-T2-hd/              # Skull-stripped IXI T2 (healthy, for Stage-1)
│   └── BraTS/                  # BraTS2020 (pathological, for evaluation)
│
├── run_pipeline.sh             # One-command full training pipeline
└── main.py                     # LDM training / evaluation entry point
```

---

## Environment

```bash
conda env create -f environment.yaml
conda activate FDP
```

`environment.yaml` pins the full dependency set (Python 3.12, PyTorch 2.10+cu130, pytorch-lightning 2.6).
The `-e .` entry at the bottom of the file installs the project itself in editable mode, so no additional `pip install` is needed.

If you prefer to install manually:

```bash
conda create -n FDP python=3.12
conda activate FDP
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install pytorch-lightning omegaconf nibabel scikit-image einops transformers taming-transformers
pip install -e .
```

Requires CUDA GPUs with ≥ 11 GB VRAM each. The pipeline uses DDP via PyTorch Lightning.
The default configuration targets 4 GPUs; see [GPU selection](#gpu-selection) to run with fewer.

---

## Datasets

- [IXI](https://brain-development.org/ixi-dataset/) — healthy T2 MRI, Stage-1 FRM training
- [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) — brain tumour, evaluation
- [BraTS2021](http://braintumorsegmentation.org/)
- [MSLUB](https://lit.fe.uni-lj.si/en/research/resources/3D-MR-MS/)
- [MSSEG-2](https://portal.fli-iam.irisa.fr/msseg-2/)

Expected layout:
```
data/IXI-T2-hd/          ← skull-stripped, flat directory, *-T2.nii.gz
data/BraTS/BraTS2020/    ← nested per-subject directories, *_t2.nii
```

---

## Training Pipeline

### Stage 0 — Preload generation (one-time prerequisite)

Before running the pipeline for the first time, generate the k-means++ prior banks that FDP needs:

```bash
python fdp/init_prior.py \
    --data_root data/IXI-T2-hd \
    --output_dir preload \
    --n_prior 128 \
    --lf_ratio 0.3
```

`--lf_ratio 0.3` stores each prior prototype at 30 % of the image size (76×76 for 256 px inputs).
This is intentionally larger than the training `mask_ratio` (default 0.10) so the same prior bank can be reused at different mask ratios without rerunning k-means.

This is a one-time step. The resulting files (`preload/t2_hf_0d3_embed_128ac_*.pth`) are consumed by Stage 1. See the [Generating / re-generating the prior banks](#generating--re-generating-the-prior-banks) section for full options.

### Stages 1–3 + Eval

Stages 1–3 and evaluation are driven by a single script. Outputs land under `logs/`.

```
logs/
└── TIMESTAMP_full_pipeline/
    ├── fdp/
    │   └── checkpoints/        last.ckpt · best.ckpt · samples_fdp.png
    ├── vae/
    │   ├── checkpoints/        last.ckpt · best.ckpt · samples_vae.png
    │   └── configs/            vae_pipeline.yaml  (generated)
    ├── ldm/
    │   ├── checkpoints/        last.ckpt · best.ckpt · samples_ldm.png
    │   └── configs/            ldm_pipeline.yaml  (generated)
    └── eval_results/
        ├── configs/            eval_pipeline.yaml (generated)
        └── eval_metrics.log
```

### Full pipeline (recommended)

```bash
bash run_pipeline.sh
```

Runs Stage-1 → Stage-2 → Stage-3 → Eval in sequence, passing checkpoints between stages automatically.

### GPU selection

GPUs can be specified either as a positional argument or via environment variable:

```bash
bash run_pipeline.sh              # default: GPUs 0,1,2,3
bash run_pipeline.sh fdp 0,1     # use GPUs 0 and 1
GPU_LIST=2,3 bash run_pipeline.sh # via env var (works with any stage)
```

### Single-stage mode

Edit the `PRETRAIN_*` paths at the top of `run_pipeline.sh` before running:

```bash
bash run_pipeline.sh fdp   # Stage-1: train FRM only
bash run_pipeline.sh vae   # Stage-2: train VAE  (requires PRETRAIN_FDP_CKPT)
bash run_pipeline.sh ldm   # Stage-3: train LDM  (requires PRETRAIN_FDP_CKPT + PRETRAIN_VAE_CKPT)
bash run_pipeline.sh eval  # Evaluation only      (requires all three PRETRAIN_* set)
```

Checkpoint dependencies per mode:

| Mode | `PRETRAIN_FDP_CKPT` | `PRETRAIN_VAE_CKPT` | `PRETRAIN_LDM_CKPT` |
|------|---------------------|---------------------|---------------------|
| full pipeline | ignored | ignored | ignored |
| `fdp` | optional (init weights) | — | — |
| `vae` | **required** | optional (resume) | — |
| `ldm` | **required** | **required** | optional (resume) |
| `eval` | **required** | **required** | optional (auto-find) |

Set to `""` to train from scratch / skip resume.

### Hyperparameters

Controlled by variables at the top of `run_pipeline.sh`.
The recommended production settings (currently commented out in the script) are:

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `FDP_MIN_EPOCHS` | 800 | Earliest epoch early stopping may trigger |
| `FDP_MAX_EPOCHS` | 1500 | Hard epoch cap |
| `FDP_BATCH` | 512 | Per-GPU batch size |
| `FDP_EARLY_STOPPING_PATIENCE` | 10 | Epochs without improvement before stopping |
| `VAE_MIN_EPOCHS` | 1000 | — |
| `VAE_MAX_EPOCHS` | 1800 | — |
| `VAE_BATCH` | 20 | Per-GPU batch size |
| `LDM_MIN_EPOCHS` | 2000 | — |
| `LDM_MAX_EPOCHS` | 3000 | — |
| `LDM_BATCH` | 256 | Per-GPU batch size |

Uncomment the first set of `*_EPOCHS` / `*_BATCH` lines in `run_pipeline.sh` and comment out the second set to switch to production settings.

### Dataset loading

All dataset loaders use **volume-level indexing** during training: one dataset item = one NIfTI volume; `__getitem__` draws a random axial slice from it.
This means epoch length ≈ number of volumes (578 for IXI T2), not number of slices.

Training datasets are **preloaded into RAM** at startup to avoid repeated disk I/O.
Validation datasets (VAE/LDM) use lazy loading from disk to avoid doubling RAM usage under DDP.

`num_workers` is set to `0` in the base YAML configs because data is already in RAM and worker fork would wastefully copy the entire cache per worker. `run_pipeline.sh` overrides this to `16` at launch time, which is safe because `ThreadPoolExecutor`-based preloading completes before the DataLoader is created.

### Stage-1 training output

```
Epoch 0000  loss=87.28  loss_real=71.18  loss_imag=15.99  loss_frec=0.00
```

| Sub-loss | Description |
|----------|-------------|
| `loss_real` | L1(predicted low-freq real, true low-freq real) |
| `loss_imag` | L1(predicted low-freq imag, true low-freq imag) |
| `loss_frec` | reserved (currently zero) |
| `loss` | `loss_real + loss_imag` |

Both the real and imaginary branches of the FRM are trained simultaneously.
At inference, only the real branch prediction is used; the imaginary component is set to zero (near-DC imaginary is theoretically negligible).

### Generating / re-generating the prior banks

`preload/` contains two k-means++ cluster-centre tensors that initialise the FRM prior context bank.
They are consumed by `configs/fdp/fdp.yaml` (`prior_path_real` / `prior_path_imag`) before Stage-1 training begins.

**When you need to (re-)generate them:**
- Fresh checkout — the `.pth` files are not tracked by git (large binaries).
- You want to adapt the prior to a different modality or dataset.
- You changed `n_prior` or `lf_ratio` in the config.

**Standard (accurate, slow):**
```bash
python fdp/init_prior.py \
    --data_root data/IXI-T2-hd \
    --output_dir preload \
    --n_prior 128 \
    --lf_ratio 0.3
```

**Fast (MiniBatchKMeans, slightly less accurate):**
```bash
python fdp/init_prior.py \
    --data_root data/IXI-T2-hd \
    --output_dir preload \
    --n_prior 128 \
    --lf_ratio 0.3 \
    --minibatch
```

All arguments and their defaults:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | *(required)* | Root of the healthy training set (e.g. `data/IXI-T2-hd`) |
| `--output_dir` | *(required)* | Where to write the `.pth` files (use `preload`) |
| `--n_prior` | `128` | Number of cluster prototypes |
| `--img_size` | `256` | Spatial size of the input slices |
| `--lf_ratio` | `0.3` | Prior storage size as a fraction of `img_size` (e.g. 0.3 → 76×76 for 256 px). Larger than training `mask_ratio` so the bank covers multiple ratios without rerunning k-means. |
| `--max_slices` | `50000` | Maximum slices sampled from the dataset |
| `--minibatch` | off | Use MiniBatchKMeans instead of KMeans |

Outputs (with defaults `n_prior=128`, `lf_ratio=0.3`):
```
preload/t2_hf_0d3_embed_128ac_real_frq.pth   # shape [128, 5776]
preload/t2_hf_0d3_embed_128ac_imag_frq.pth   # shape [128, 5776]
```
These paths are already wired into `configs/fdp/fdp.yaml`; no config change is needed as long as you keep the defaults.

---

## Visual Sampling

`scripts/sample_vis.py` generates a PNG grid from any checkpoint for quick quality checks. It is run automatically at the end of each stage by `run_pipeline.sh`.

### FDP — `[Input | frec | hf_supp]`

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample_vis.py fdp \
    --ckpt logs/<run>/fdp/checkpoints \
    --out_dir logs/<run>/fdp \
    --n 6
```

### VAE — `[Input | VAE(Input) | |Diff| | frec | VAE(frec) | hf_supp | VAE(hf_supp)]`

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample_vis.py vae \
    --ckpt logs/<run>/vae/checkpoints/last.ckpt \
    --config logs/<run>/vae/configs/vae_pipeline.yaml \
    --out_dir logs/<run>/vae \
    --n 6
```

### LDM — `[Input | frec | LDM-rec | |Diff|]`

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample_vis.py ldm \
    --ckpt logs/<run>/ldm/checkpoints \
    --config logs/<run>/ldm/configs/ldm_pipeline.yaml \
    --out_dir logs/<run>/ldm \
    --n 6
```

Passing a **directory** as `--ckpt` automatically selects `best.ckpt` → `last.ckpt`.

### Switching datasets

All three modes accept `--data_root` to override the default (`data/IXI-T2-hd`).
To run on **BraTS** instead:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample_vis.py ldm \
    --ckpt logs/<run>/ldm/checkpoints/last.ckpt \
    --config logs/<run>/ldm/configs/ldm_pipeline.yaml \
    --out_dir ./tmp \
    --data_root data/BraTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
```

The script scans `--data_root` recursively for `.nii` / `.nii.gz` files, so the nested per-subject layout of BraTS is handled automatically.  
Note that models are **trained on healthy IXI T2**; BraTS inference is for anomaly-detection evaluation only — the residual map `|Input − LDM-rec|` is the anomaly score.

---

## Using FDPModule in Your Own Code

```python
import torch
from fdp import FDPModule

fdp = FDPModule.from_checkpoint("logs/<run>/fdp/checkpoints/best.ckpt").eval().cuda()

x = ...  # (N, 1, 256, 256) MRI slice tensor in [-1, 1]

with torch.no_grad():
    frec, hf_supp = fdp(x)
    # frec    — frequency-reconstructed "healthy" MRI  (low-freq conditioning)
    # hf_supp — high-frequency supplement              (structural conditioning)

# Post-processing after your generative model produces a reconstruction:
result = fdp.enhance(x, reconstruction)  # amplitude-alignment
```

---

## FDP Pipeline

```
Input MRI  x
    │
    ├─ FFT → low-freq patch fl  ──→  FRM cross-attention (prior bank P) → f̂l
    │                                    loss_real + loss_imag  ← Stage-1 loss
    │
    ├─ MERGE(f̂l, original high-freq fh) → IDFT → frec     (low-freq conditioning)
    │
    └─ High-pass filter → hf_supp                           (high-freq conditioning)

LDM(frec_latent ⊕ hf_supp_latent) → reconstruction

Post-processing: enhance(x, reconstruction) → final output
Residual map:   |x - final output|           → anomaly score
```

---

## Configuration Reference

### `configs/fdp/fdp.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `fdp.n_prior` | 128 | Number of prior context prototypes |
| `fdp.mask_ratio` | 0.10 | Low-freq region size (fraction of image); the extracted patch is `(img_size × mask_ratio)²` |
| `fdp.img_size` | 256 | Input spatial size |
| `fdp.dropout` | 0.2 | Dropout rate applied in attention weights and conv block (regularisation) |
| `fdp.prior_path_real` | `preload/t2_hf_0d3_embed_128ac_real_frq.pth` | Pre-trained real prior |
| `fdp.prior_path_imag` | `preload/t2_hf_0d3_embed_128ac_imag_frq.pth` | Pre-trained imag prior |
| `train.lr` | 2e-5 | Learning rate |
| `train.epochs` | 800 | Max epochs (overridden by `FDP_MAX_EPOCHS` in `run_pipeline.sh`) |
| `train.batch_size` | 8 | Batch size (overridden by `FDP_BATCH` in `run_pipeline.sh`) |
| `train.num_workers` | 0 | DataLoader workers; 0 is correct since data is preloaded into RAM (`run_pipeline.sh` overrides to 16) |

---

## Acknowledgements

We thank [Claude](https://claude.ai) (Anthropic) for assistance with code organisation and documentation.

---

## Citation

```bibtex
@inproceedings{li2026fdp,
  title={FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI},
  author={Li, Hao and Zhuang, Zhenfeng and Lin, Jingyu and Liu, Yu and Chen, Yifei and Peng, Qiong and Yu, Lequan and Wang, Liansheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={8},
  pages={6118--6126},
  year={2026}
}
```
