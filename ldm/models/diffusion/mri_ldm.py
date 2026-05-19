import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import erfinv as _erf_inv
from tqdm import tqdm
from .ddpm import LatentDiffusion
from ldm.modules.metrics.test_utils import (
    test_metrics, arverage_metric, filter_3d_connected_components, dice)


class MRILatentDiffusion(LatentDiffusion):
    """
    LDM subclass for brain MRI anomaly detection.

    Stage-1 (FRM training) is handled entirely by fdp/train.py.
    This class is used for Stage-2 (diffusion model training) and inference.

    The cond_stage_model should be an FDPModule (fdp.module.FDPModule) loaded
    from a Stage-1 checkpoint via ckpt_path in the config.
    get_learned_conditioning encodes x → (frec, hf_supp) → VAE latents → concat.
    """

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        saved_path=None,
        k=None,
        ddim=True,
        ddim_steps=50,
        eta=1.0,
        *args,
        **kwargs,
    ):
        # Capture FDP ckpt path before super().__init__ so _reload_fdp works
        # after init_from_ckpt (which may overwrite FDP weights from the LDM ckpt).
        if hasattr(cond_stage_config, "params"):
            self._fdp_ckpt_path = getattr(cond_stage_config.params, "ckpt_path", None)
        elif isinstance(cond_stage_config, dict):
            self._fdp_ckpt_path = cond_stage_config.get("params", {}).get("ckpt_path")
        else:
            self._fdp_ckpt_path = None

        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            *args,
            **kwargs,
        )
        self.saved_path = saved_path
        if saved_path:
            os.makedirs(saved_path, exist_ok=True)
        self.k = k
        self.ddim = ddim
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.val_bestThresh = 0
        self.metric_func = test_metrics
        self.metric_mean_func = arverage_metric
        self.val_mri_metric  = {k: [] for k in ("AUPRC", "AUROC", "bestThresh", "bestDice", "diceScore", "dice_at_percentiles")}
        self.test_mri_metric = {k: [] for k in ("AUPRC", "AUROC", "bestThresh", "bestDice", "diceScore", "dice_at_percentiles")}

    # ------------------------------------------------------------------
    # FDP hot-swap: reload FDP from config ckpt_path after any checkpoint
    # restore, so a new FDP can be used with an existing LDM checkpoint.
    # ------------------------------------------------------------------

    def _reload_fdp(self) -> None:
        path = getattr(self, "_fdp_ckpt_path", None)
        if not path:
            return
        from fdp.module import _extract_fdp_state
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cond_stage_model.load_state_dict(_extract_fdp_state(ckpt))
        print(f"[MRILatentDiffusion] FDP reloaded from {path}")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        # Load full LDM checkpoint (including any embedded FDP weights), then
        # override FDP with the path from config if one was specified.
        super().init_from_ckpt(path, ignore_keys=ignore_keys, only_model=only_model)
        self._reload_fdp()

    def on_train_start(self) -> None:
        # Handles Lightning resume path (trainer.fit ckpt_path=...),
        # where load_state_dict runs before this hook.
        self._reload_fdp()

    def on_test_start(self) -> None:
        self._reload_fdp()

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def get_learned_conditioning(self, c):
        """
        Run FDPModule on raw MRI slice c, encode outputs through the VAE,
        and return the concatenated latent conditioning tensor.

        Args:
            c: (N, 1, H, W) MRI slice in [-1, 1]
        Returns:
            (N, 8, h, w) — frec_latent cat hf_supp_latent
        """
        with torch.no_grad():
            frec, hf_supp = self.cond_stage_model(c)
        c_frec = self.get_first_stage_encoding(self.encode_first_stage(frec)).detach()
        c_hf   = self.get_first_stage_encoding(self.encode_first_stage(hf_supp)).detach()
        return torch.cat([c_frec, c_hf], dim=1)

    # ------------------------------------------------------------------
    # Data input
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False,
                  force_c_encode=False, cond_key=None, return_original_cond=False,
                  bs=None):
        x = super(LatentDiffusion, self).get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            cond_key = cond_key or self.cond_stage_key
            xc = x if cond_key == self.first_stage_key else super().get_input(batch, cond_key).to(self.device)
            if not self.cond_stage_trainable or force_c_encode:
                c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
        else:
            c = None
            xc = None

        out = [z, c]
        if return_first_stage_outputs:
            out.extend([x, self.decode_first_stage(z)])
        if return_original_cond:
            out.append(xc)
        return out

    # ------------------------------------------------------------------
    # Validation / test
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        data = batch[self.first_stage_key]
        ind = data.shape[-1] // 2
        batch[self.first_stage_key] = data[..., ind:ind + 1]
        super().validation_step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch, _batch_idx):
        data = batch[self.first_stage_key]
        ID = batch["file_path"]
        GT_volume = batch.get("seg", None)

        rec_list, mask_list = [], []
        n_slices = data.shape[-1]
        _pid = os.path.basename(os.path.dirname(ID[0])) if ID else ''
        for ind in tqdm(range(n_slices), desc=f"Eval {_pid}", total=n_slices, leave=True):
            cur_x_start = data[..., ind:ind + 1]
            batch[self.first_stage_key] = cur_x_start

            z, c = self.get_input(batch, self.first_stage_key, force_c_encode=True)
            samples, _ = self.sample_log(
                cond=c, batch_size=data.shape[0], ddim=self.ddim,
                ddim_steps=self.ddim_steps, eta=self.eta, x0=z,
                mask=None, noise_dropout=0.0)

            x_samples = self.cond_stage_model.enhance(
                cur_x_start.permute(0, 3, 1, 2),
                self.decode_first_stage(samples),
            ).permute(0, 2, 3, 1).cpu()

            x_samples = torch.where(
                cur_x_start.cpu().float() > -1.0,
                x_samples,
                torch.tensor(-1.0),
            ).clamp(-1.0, 1.0)

            _x_cpu = cur_x_start.float().cpu()
            mask = torch.where(
                _x_cpu > -1.0,
                torch.ones_like(_x_cpu),
                torch.zeros_like(_x_cpu),
            )
            mask_list.append(mask)
            rec_list.append(x_samples)

        if rec_list:
            rec_volume  = torch.cat(rec_list,  dim=-1)
            mask_volume = torch.cat(mask_list, dim=-1)
            diff_volume = torch.abs(data.detach().cpu() - rec_volume)

            self.save_volumes(diff_volume, mask_volume, data.detach().cpu(),
                              rec_volume, GT_volume, ID,
                              vis_save_path=self.saved_path)
            # Run metrics first — test_metrics runs the 3D pipeline once and
            # saves the thresholded prediction to res/*.pt.
            self.get_test_metric(
                ID,
                vis_save_path=self.saved_path,
                diff_volume=diff_volume,
                mask_volume=mask_volume,
                data_seg=GT_volume,
            )
            # Extract per-patient thresholds from what test_metrics just used.
            if self.val_bestThresh:
                batch_thrs = [self.val_bestThresh] * len(ID)
            elif self.test_mri_metric["bestThresh"]:
                batch_thrs = self.test_mri_metric["bestThresh"][-len(ID):]
            else:
                batch_thrs = [None] * len(ID)
            # Visualize using the res/*.pt files saved by test_metrics.
            self.save_eval_vis(data.detach().cpu(), rec_volume, diff_volume,
                               GT_volume, ID, batch_thrs=batch_thrs,
                               vis_save_path=self.saved_path)

    def save_eval_vis(self, data, rec_volume, diff_volume,
                      GT_volume, ID, batch_thrs, vis_save_path, n_rows: int = 10):
        """Save a grid image (n_rows × cols) for each patient in the batch.
        Loads pred from res/*.pt saved by test_metrics — same result used for eval."""
        out_dir = os.path.join(vis_save_path, "vis")
        os.makedirs(out_dir, exist_ok=True)

        has_seg = GT_volume is not None
        # columns: Input, LDM-rec, |Diff|, Pred, GroundTruth
        ncols = 5

        for i in range(len(ID)):
            stem = os.path.basename(ID[i]).replace(".nii.gz", "").replace(".nii", "")

            # Load the exact prediction used by test_metrics (thresholded + CC-filtered)
            res_name = (os.path.basename(ID[i])
                        .replace("t2", "res")
                        .replace(".nii.gz", ".pt")
                        .replace(".nii", ".pt"))
            res_path = os.path.join(vis_save_path, "res", res_name)
            if os.path.exists(res_path):
                pred_vol = np.squeeze(torch.load(res_path, map_location="cpu",
                                                  weights_only=False).float().numpy())
            else:
                pred_vol = np.zeros(data.shape[1:3] + (data.shape[-1],), dtype=np.float32)

            thr = batch_thrs[i] if i < len(batch_thrs) else None
            thr_title = f"Pred ({thr:.4g})" if thr is not None else "Pred (n/a)"
            # column order: Input, LDM-rec, |Diff|, Pred, GroundTruth
            col_titles = ["Input", "LDM-rec", "|Diff|", thr_title, "GroundTruth"]

            n_slices = data.shape[-1]
            # Sample slice indices with Gaussian density (more from centre, fewer from edges)
            mu = (n_slices - 1) / 2.0
            sigma = n_slices / 5.0
            quantiles = np.linspace(0.025, 0.975, n_rows)
            indices = np.clip(
                (mu + sigma * np.sqrt(2) * np.array([
                    _erf_inv(2 * q - 1) for q in quantiles
                ])).round().astype(int),
                0, n_slices - 1,
            )

            _, axes = plt.subplots(n_rows, ncols,
                                   figsize=(ncols * 2.4, n_rows * 2.4),
                                   squeeze=False)
            for r, ind in enumerate(indices):
                inp  = data[i, :, :, ind].cpu().numpy()
                rec  = rec_volume[i, :, :, ind].cpu().numpy()
                diff = diff_volume[i, :, :, ind].cpu().numpy()
                pred = pred_vol[:, :, ind].astype(np.float32)
                seg  = (GT_volume[i, :, :, ind].cpu().numpy().astype(np.float32)
                        if has_seg else np.zeros_like(inp))

                cols = [(inp, False), (rec, False), (diff, True), (pred, True), (seg, True)]

                for c, (img, is_diff) in enumerate(cols):
                    ax = axes[r][c]
                    if is_diff:
                        ax.imshow(np.abs(img), cmap="hot", vmin=0, vmax=1)
                    else:
                        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
                    ax.axis("off")
                    if r == 0:
                        ax.set_title(col_titles[c], fontsize=11, fontweight="bold")

            plt.tight_layout(pad=0.3)
            plt.savefig(os.path.join(out_dir, f"{stem}.jpg"), dpi=80)
            plt.close()

    def save_volumes(self, diff_volume, mask_volume, data, rec_volume,
                     GT_volume, ID, vis_save_path):
        for i in range(len(ID)):
            stem = os.path.basename(ID[i]).replace(".nii.gz", "").replace(".nii", "")
            for sub, tensor in [("mask",   mask_volume[i]),
                                 ("diff",   diff_volume[i]),
                                 ("input",  data[i]),
                                 ("rec",    rec_volume[i])]:
                os.makedirs(os.path.join(vis_save_path, sub), exist_ok=True)
                torch.save(tensor, os.path.join(vis_save_path, sub, f"{stem}_{sub}.pt"))
            if GT_volume is not None:
                os.makedirs(os.path.join(vis_save_path, "seg"), exist_ok=True)
                torch.save(GT_volume[i].detach().cpu().bool(),
                           os.path.join(vis_save_path, "seg", f"{stem}_seg.pt"))

    # column widths for the result table
    _COL_ID   = 30
    _COL_SEP  = "─" * 84

    def _metric_header(self):
        h = (f"{'#':>4}  {'Patient':<{self._COL_ID}}  {'AUPRC':>6}  {'AUROC':>6}"
             f"  {'Dice':>6}  {'bestDice':>8}  {'Threshold':>9}")
        tqdm.write("\n" + self._COL_SEP)
        tqdm.write(h)
        tqdm.write(self._COL_SEP)
        with open(os.path.join(self.saved_path, "eval_metrics.log"), "a") as f:
            f.write(self._COL_SEP + "\n" + h + "\n" + self._COL_SEP + "\n")

    def get_test_metric(self, ID, vis_save_path,
                        diff_volume, mask_volume, data_seg):
        if data_seg is None:
            return
        for i in range(len(ID)):
            self.test_mri_metric = self.metric_func(
                ID[i], vis_save_path, self.test_mri_metric,
                data_seg[i].cpu(), diff_volume[i], mask_volume[i],
                val_bestThresh=self.val_bestThresh)

            n_done = len(self.test_mri_metric["AUPRC"])
            pat_auprc    = self.test_mri_metric["AUPRC"][-1]
            pat_auroc    = self.test_mri_metric["AUROC"][-1]
            pat_dice     = self.test_mri_metric["diceScore"][-1]
            pat_bestdice = self.test_mri_metric["bestDice"][-1]
            pat_thresh   = (self.val_bestThresh if self.val_bestThresh
                            else self.test_mri_metric["bestThresh"][-1])
            _id_str = os.path.basename(os.path.dirname(ID[i]))

            if n_done == 1:
                self._metric_header()

            row = (f"[{n_done:>2}]  {_id_str:<{self._COL_ID}}"
                   f"  {pat_auprc:.4f}  {pat_auroc:.4f}  {pat_dice:.4f}  {pat_bestdice:.4f}"
                   f"  {pat_thresh:.6f}")
            tqdm.write(row)
            with open(os.path.join(self.saved_path, "eval_metrics.log"), "a") as f:
                f.write(row + "\n")

        metrics = self.metric_mean_func(self.test_mri_metric, self.val_bestThresh)
        mean_thresh = (self.val_bestThresh if self.val_bestThresh
                       else metrics.get("bestThreshMean", float("nan")))
        mean_row = (f"{'mean':>38}"
                    f"  {metrics['AUPRCMean']:.4f}  {metrics['AUROCMean']:.4f}"
                    f"  {metrics['diceScoreMean']:.4f}  {metrics['bestDiceMean']:.4f}"
                    f"  {mean_thresh:.6f}")
        tqdm.write(self._COL_SEP)
        tqdm.write(mean_row)
        tqdm.write(self._COL_SEP)
        with open(os.path.join(self.saved_path, "eval_metrics.log"), "a") as f:
            f.write(self._COL_SEP + "\n" + mean_row + "\n" + self._COL_SEP + "\n")

    def on_test_epoch_end(self):
        if not self.test_mri_metric["AUPRC"]:
            return
        metrics = self.metric_mean_func(self.test_mri_metric, self.val_bestThresh)

        # ── Global percentile threshold ──────────────────────────────────────
        # Find the single percentile level that maximises mean Dice across all
        # patients (computed without CC filter during test_metrics), then
        # re-evaluate every patient with that percentile + adaptive CC filter.
        global_dice_mean = 0.0
        best_pct = 0.0
        dap = self.test_mri_metric.get("dice_at_percentiles", [])
        if self.saved_path and dap:
            dap_matrix = np.array(dap)                        # (N_pat, 200)
            mean_per_pct = dap_matrix.mean(axis=0)            # (200,)
            best_pct_idx = int(np.argmax(mean_per_pct))
            best_pct = float(np.linspace(0.5, 99.5, 200)[best_pct_idx])

            proc_diff_dir = os.path.join(self.saved_path, 'proc_diff')
            seg_dir = os.path.join(self.saved_path, 'seg')
            global_dices = []
            for proc_f in sorted(glob.glob(os.path.join(proc_diff_dir, '*.pt'))):
                stem = os.path.basename(proc_f).replace('_proc_diff.pt', '')
                seg_f = os.path.join(seg_dir, f"{stem}_seg.pt")
                if not os.path.exists(seg_f):
                    continue
                diff_np = torch.load(proc_f, weights_only=True).numpy()
                seg_v = torch.load(seg_f, weights_only=True).numpy().flatten().astype(bool)
                thresh = float(np.percentile(diff_np.flatten(), best_pct))
                pred = diff_np > thresh
                pred_cc = filter_3d_connected_components(
                    np.squeeze(pred).copy(), min_area=8)
                global_dices.append(dice(pred_cc.flatten(), seg_v))

            if global_dices:
                global_dice_mean = float(np.mean(global_dices))

        # ── Summary ──────────────────────────────────────────────────────────
        sep = "═" * 72
        summary = (f"\n{sep}\n"
                   f"  FINAL ({len(self.test_mri_metric['AUPRC'])} patients)\n"
                   f"  AUPRC={metrics['AUPRCMean']:.4f}  "
                   f"AUROC={metrics['AUROCMean']:.4f}  "
                   f"OracleDice={metrics['diceScoreMean']:.4f}  "
                   f"bestDice={metrics['bestDiceMean']:.4f}\n"
                   f"  GlobalDice={global_dice_mean:.4f}  (percentile={best_pct:.1f}%)\n"
                   f"{sep}")
        tqdm.write(summary)
        if self.saved_path:
            with open(os.path.join(self.saved_path, "eval_metrics.log"), "a") as f:
                f.write(summary + "\n")
