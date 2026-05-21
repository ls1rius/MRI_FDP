"""
Stage-0 training: optimise the FDP Frequency Reconstruction Module (FRM).

Single GPU:
    python fdp/train.py --config configs/fdp/fdp.yaml

Multi-GPU (Lightning DDP, no torchrun needed):
    python fdp/train.py --config configs/fdp/fdp.yaml --devices 4
"""
import argparse
import math
import os
import shutil
import sys
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

class EarlyStoppingAfterMinEpochs(EarlyStopping):
    """EarlyStopping whose patience counter only starts after min_epochs is reached."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._past_min = False

    def _reset_on_min_cross(self, trainer) -> bool:
        if trainer.current_epoch < trainer.min_epochs:
            return False
        if not self._past_min:
            self._past_min = True
            self.wait_count = 0
            # best_score intentionally NOT reset: PL initialises it to torch_inf/-torch_inf
            # (not None), so resetting to None breaks .to() in _evaluate_stopping_criteria.
        return True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._reset_on_min_cross(trainer):
            super().on_validation_epoch_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._reset_on_min_cross(trainer):
            super().on_train_epoch_end(trainer, pl_module)
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdp.module import FDPModule
from fdp.data import MRIDataset
from fdp.data.dataset import _scan_t2_files


def _collate(batch: list) -> torch.Tensor:
    imgs = [torch.from_numpy(item["image"]).float() for item in batch]
    return torch.stack([img.permute(2, 0, 1) for img in imgs])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class FDPDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: dict, batch_size: int, num_workers: int):
        super().__init__()
        self.data_cfg    = data_cfg
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = MRIDataset(
            data_root=self.data_cfg["train_root"],
            size=self.data_cfg["size"],
            is_test=False,
            random_sample=self.data_cfg.get("random_sample", True),
        )
        val_root = self.data_cfg.get("val_root")
        self.val_dataset = MRIDataset(
            data_root=val_root,
            size=self.data_cfg["size"],
            is_test=False,
            random_sample=False,
        ) if val_root else None
        if self.val_dataset is not None and len(self.val_dataset) == 0:
            rank_zero_info(
                f"Validation root '{val_root}' produced 0 samples; "
                "disabling validation and monitoring train/loss."
            )
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=True,
            drop_last=False,
        )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class FDPLightningModule(pl.LightningModule):
    def __init__(self, fdp_cfg: dict, lr: float = 2e-5,
                 min_lr: float = 1e-7, warmup_epochs: int = 10,
                 scheduler: str = "sgdr", cyclic_step_epochs: int = 50,
                 sgdr_t0: int = 50, sgdr_t_mult: int = 2,
                 max_epochs: int = 800):
        super().__init__()
        self.save_hyperparameters()
        self.model = FDPModule(
            n_prior=fdp_cfg["n_prior"],
            mask_ratio=fdp_cfg["mask_ratio"],
            img_size=fdp_cfg["img_size"],
            prior_path_real=fdp_cfg.get("prior_path_real"),
            prior_path_imag=fdp_cfg.get("prior_path_imag"),
        )

    def training_step(self, batch, batch_idx):
        _, d = self.model.compute_loss(batch)
        self.log("train/loss",      d["loss"],      on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/loss_real", d["loss_real"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_imag", d["loss_imag"], on_step=False, on_epoch=True, sync_dist=True)
        return d["loss"]

    def validation_step(self, batch, batch_idx):
        _, d = self.model.compute_loss(batch)
        self.log("val/loss",      d["loss"],      on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/loss_real", d["loss_real"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_imag", d["loss_imag"], on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        sched_name = self.hparams.scheduler

        if sched_name == "none":
            return opt

        if sched_name == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=5,
                min_lr=self.hparams.min_lr,
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "monitor": "val/loss",
                                     "interval": "epoch", "frequency": 1}}

        if sched_name == "sgdr":
            # CosineAnnealingWarmRestarts: smooth cosine with warm restarts,
            # period doubles each restart (T_mult=2): 50→100→200→400…
            # Explores broadly early on, converges stably in long final cycle.
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=self.hparams.sgdr_t0,
                T_mult=self.hparams.sgdr_t_mult,
                eta_min=self.hparams.min_lr,
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}

        if sched_name == "cyclic":
            step_size = self.hparams.cyclic_step_epochs
            sched = torch.optim.lr_scheduler.CyclicLR(
                opt,
                base_lr=self.hparams.min_lr,
                max_lr=self.hparams.lr,
                step_size_up=step_size,
                mode="triangular2",
                cycle_momentum=False,
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}

        # cosine: linear warmup then cosine anneal
        warmup = self.hparams.warmup_epochs
        total  = self.hparams.max_epochs

        def _lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_frac = self.hparams.min_lr / self.hparams.lr
            return min_frac + (1.0 - min_frac) * cos

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}

    def on_save_checkpoint(self, checkpoint):
        # embed FDP-format keys so FDPModule loaders work with this ckpt directly
        checkpoint["fdp_model"] = self.model.state_dict()
        checkpoint["fdp_config"] = {
            "n_prior":    self.hparams.fdp_cfg["n_prior"],
            "mask_ratio": self.hparams.fdp_cfg["mask_ratio"],
            "img_size":   self.hparams.fdp_cfg["img_size"],
        }


# ---------------------------------------------------------------------------
# Epoch logger (same style as main.py EpochLossLogger)
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class EpochLogger(Callback):
    def __init__(self, monitor: str):
        super().__init__()
        self.monitor = monitor
        self.best_value = None

    def on_fit_start(self, trainer, pl_module):
        self.fit_start_time = time.time()

    def _append_best(self, parts, metrics):
        if self.monitor not in metrics:
            return
        try:
            current = float(metrics[self.monitor])
        except Exception:
            return
        if self.best_value is None or current < self.best_value:
            self.best_value = current
        parts.append(f"best/{self.monitor}={self.best_value:.4f}")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        parts = [f"global_step={trainer.global_step}"]
        elapsed = time.time() - getattr(self, "fit_start_time", time.time())
        completed = epoch + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        parts.append(f"elapsed={_format_duration(elapsed)}")
        if max_epochs and max_epochs > 0 and completed > 0:
            remaining = max(0, max_epochs - completed)
            eta = elapsed / completed * remaining
            parts.append(f"eta={_format_duration(eta)}")
        try:
            opt = pl_module.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            parts.append(f"lr={opt.param_groups[0]['lr']:.2e}")
        except Exception:
            pass
        for k in sorted(trainer.callback_metrics):
            if "lr" in k.lower():
                continue
            try:
                parts.append(f"{k}={float(trainer.callback_metrics[k]):.4f}")
            except Exception:
                pass
        self._append_best(parts, trainer.callback_metrics)
        rank_zero_info(f"[Epoch {epoch:04d}] " + "  ".join(parts))


class BestCheckpointAlias(Callback):
    def __init__(self, filename: str = "best.ckpt"):
        super().__init__()
        self.filename = filename
        self._last_source = None

    def _sync_best(self, trainer) -> None:
        if not trainer.is_global_zero:
            return
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_path = cb.best_model_path
                if not best_path or best_path == self._last_source or not os.path.exists(best_path):
                    return
                alias_path = os.path.join(cb.dirpath, self.filename)
                if os.path.abspath(best_path) == os.path.abspath(alias_path):
                    return
                shutil.copy2(best_path, alias_path)
                self._last_source = best_path
                rank_zero_info(f"Updated best checkpoint alias: {alias_path} -> {best_path}")
                return

    def on_train_epoch_end(self, trainer, pl_module):
        self._sync_best(trainer)

    def on_validation_end(self, trainer, pl_module):
        self._sync_best(trainer)

    def on_fit_end(self, trainer, pl_module):
        self._sync_best(trainer)


def _save_effective_config(cfg: dict, out_dir: str) -> None:
    """Save the actual FDP config after CLI overrides, matching pipeline logs."""
    if os.environ.get("LOCAL_RANK", "0") not in ("0", ""):
        return
    cfg_dir = os.path.join(out_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, "fdp_pipeline.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    rank_zero_info(f"Saved effective FDP config to {path}")


def _has_validation_data(data_cfg: dict) -> bool:
    val_root = data_cfg.get("val_root")
    if not val_root:
        return False
    matches = _scan_t2_files(val_root)
    if not matches:
        rank_zero_info(
            f"Validation root '{val_root}' produced 0 samples; "
            "disabling validation and monitoring train/loss."
        )
        data_cfg["val_root"] = None
        return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/fdp/fdp.yaml")
    parser.add_argument("--train_root",  default=None)
    parser.add_argument("--min_epochs",  type=int,   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--num_workers", type=int,   default=None)
    parser.add_argument("--val_every",   type=int,   default=None)
    parser.add_argument("--early_stopping_patience",  type=int,   default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--out_dir",  default=None)
    parser.add_argument("--pretrain", default=None,
                        help="load FDP weights (epoch resets to 0)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore existing last.ckpt and train from scratch")
    parser.add_argument("--devices", type=int, default=None,
                        help="number of GPUs (default: all available, max 4)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    data_cfg  = cfg["data"]
    fdp_cfg   = cfg["fdp"]

    if args.train_root  is not None: data_cfg["train_root"]  = args.train_root
    if args.min_epochs  is not None: train_cfg["min_epochs"] = args.min_epochs
    if args.epochs      is not None: train_cfg["epochs"]     = args.epochs
    if args.batch_size  is not None: train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None: train_cfg["num_workers"] = args.num_workers
    if args.val_every   is not None: train_cfg["val_every"]  = args.val_every
    if args.early_stopping_patience  is not None:
        train_cfg["early_stopping_patience"]  = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        train_cfg["early_stopping_min_delta"] = args.early_stopping_min_delta
    if args.out_dir is not None:
        train_cfg["output_dir"] = args.out_dir

    out_dir  = train_cfg.get("output_dir", "./outputs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    # ── Model ────────────────────────────────────────────────────────────────
    pl_model = FDPLightningModule(
        fdp_cfg=fdp_cfg,
        lr=train_cfg["lr"],
        min_lr=train_cfg.get("min_lr", 1e-7),
        warmup_epochs=train_cfg.get("warmup_epochs", 10),
        scheduler=train_cfg.get("scheduler", "sgdr"),
        cyclic_step_epochs=train_cfg.get("cyclic_step_epochs", 50),
        sgdr_t0=train_cfg.get("sgdr_t0", 50),
        sgdr_t_mult=train_cfg.get("sgdr_t_mult", 2),
        max_epochs=train_cfg["epochs"],
    )

    if args.pretrain:
        ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        state = ckpt.get("fdp_model") or ckpt.get("model")
        if state:
            pl_model.model.load_state_dict(state)
            rank_zero_info(f"Loaded pretrained FDP weights from {args.pretrain}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dm = FDPDataModule(data_cfg, train_cfg["batch_size"], train_cfg.get("num_workers", 4))

    has_val = _has_validation_data(data_cfg)
    monitor  = "val/loss" if has_val else "train/loss"
    check_every = train_cfg.get("val_every", train_cfg.get("save_every", 5))
    val_every = check_every if has_val else 999999
    _save_effective_config(cfg, out_dir)
    rank_zero_info(
        f"Checkpoint monitor: {monitor}; "
        f"{'validation/checkpoint interval' if has_val else 'checkpoint interval'}="
        f"{check_every}"
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:06d}",
            monitor=monitor,
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
            every_n_epochs=check_every,
            save_on_train_epoch_end=not has_val,
        ),
        EarlyStoppingAfterMinEpochs(
            monitor=monitor,
            mode="min",
            patience=train_cfg.get("early_stopping_patience", 10),
            min_delta=train_cfg.get("early_stopping_min_delta", 1e-4),
            verbose=False,
            check_on_train_epoch_end=not has_val,
        ),
        EpochLogger(monitor=monitor),
        BestCheckpointAlias(),
    ]

    # ── Trainer ──────────────────────────────────────────────────────────────
    n_gpu    = torch.cuda.device_count()
    devices  = args.devices or min(n_gpu, 4) or 1
    # used in the training forward pass.
    strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"

    trainer = pl.Trainer(
        default_root_dir=out_dir,
        max_epochs=train_cfg["epochs"],
        min_epochs=train_cfg.get("min_epochs", 0),
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=50,
        check_val_every_n_epoch=val_every,
        num_sanity_val_steps=0,
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    resume_ckpt = None
    last_ckpt   = os.path.join(ckpt_dir, "last.ckpt")
    if not args.no_resume and os.path.exists(last_ckpt):
        resume_ckpt = last_ckpt
        rank_zero_info(f"Resuming from {last_ckpt}")

    trainer.fit(pl_model, datamodule=dm, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
