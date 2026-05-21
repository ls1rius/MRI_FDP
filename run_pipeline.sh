#!/bin/bash
# =============================================================================
# FDP Full Training Pipeline
#
# Stage 0 – Preload  : generate prior banks (ONE-TIME prerequisite, not automated)
#               python fdp/init_prior.py --data_root data/IXI-T2-hd \
#                                        --output_dir preload --n_prior 128 \
#                                        --lf_ratio 0.3
#             lf_ratio=0.3 → stores 76×76 proto per cluster (30% of 256px).
#             Must be ≥ mask_ratio used in training (default 0.10); runtime
#             _crop_prior() crops the centre mask_ratio region from each proto.
#
#   Stage 1 – FDP   : 100 epochs, 4×GPU (DDP),          batch 512/GPU → 2048 total
#   Stage 2 – VAE   : 500 epochs, 4×GPU (DDP),          batch  20/GPU →   80 total
#   Stage 3 – LDM   : 800 epochs, 4×GPU (DDP),          batch 256/GPU → 1024 total
#
# All outputs land under logs/:
#   Full pipeline  → logs/TIMESTAMP_full_pipeline/{fdp,vae,ldm}/
#   Single stage   → logs/TIMESTAMP_<stage>_pipeline/<stage>/
#
# Usage:
#   bash run_pipeline.sh                  # run all four stages (FDP→VAE→LDM→eval), GPUs 0,1,2,3
#   bash run_pipeline.sh fdp              # run only FDP stage
#   bash run_pipeline.sh vae              # run only VAE stage
#   bash run_pipeline.sh ldm             # run only LDM stage  (uses PRETRAIN_* below)
#   bash run_pipeline.sh eval            # run only evaluation (uses PRETRAIN_* below)
#
# Specify GPUs (default: 0,1,2,3):
#   bash run_pipeline.sh [stage] 0,1      # use GPU 0 and 1
#   GPU_LIST=2,3 bash run_pipeline.sh     # via env var
#
# PRETRAIN_* checkpoint dependencies per mode:
#
#   Mode                  PRETRAIN_FDP_CKPT   PRETRAIN_VAE_CKPT   PRETRAIN_LDM_CKPT
#   ──────────────────    ─────────────────   ─────────────────   ─────────────────
#   bash run_pipeline.sh  (ignored – pipeline passes ckpts between stages directly)
#   bash ... fdp          optional: if set, load as init weights; else train from scratch
#   bash ... vae          required            optional: if set, resume from it; else auto-find/scratch
#   bash ... ldm          required            required            optional: if set, resume from it; else auto-find/scratch
#   bash ... eval         required            required            optional: if set, use it;
#                                                                 else auto-find latest LDM ckpt
#
#   "required" → file must exist or script exits with error
#   "optional" → set to "" to skip; behaviour described above
# =============================================================================
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PYTHON="python -u"
ROOT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
cd "$ROOT"

LOGS="$ROOT/logs"
mkdir -p "$LOGS"

# ── Logging helper ────────────────────────────────────────────────────────────
# LOGFILE is overridden once RUN_DIR is known (see below).
LOGFILE="$LOGS/pipeline.log"
log() {
    local msg="[$(date '+%F %T')] $*"
    echo "$msg" | tee -a "$LOGFILE"
}
die() { log "ERROR: $*"; exit 1; }

# ── Signal handling ───────────────────────────────────────────────────────────
_shutdown() {
    trap - INT TERM
    log "Interrupted — killing process group $$"
    kill -- -$$ 2>/dev/null || true
}
trap '_shutdown' INT TERM

export PYTHONUNBUFFERED=1

# ── Hyperparameters ───────────────────────────────────────────────────────────
FDP_MIN_EPOCHS=200  ;  FDP_MAX_EPOCHS=300   ;  FDP_BATCH=8  ;  FDP_VAL_EVERY=5  ;  FDP_EARLY_STOPPING_PATIENCE=10
VAE_MIN_EPOCHS=800  ;  VAE_MAX_EPOCHS=1000  ;  VAE_BATCH=8  ;  VAE_VAL_EVERY=5  ;  VAE_EARLY_STOPPING_PATIENCE=10
LDM_MIN_EPOCHS=600  ;  LDM_MAX_EPOCHS=800   ;  LDM_BATCH=8  ;  LDM_VAL_EVERY=5  ;  LDM_EARLY_STOPPING_PATIENCE=10
VAL_MAX_SAMPLES=32

# ── Pretrained checkpoints (used when running a single stage) ────────────────
# Update these before running a standalone stage.
PRETRAIN_FDP_CKPT=""
PRETRAIN_VAE_CKPT=""
PRETRAIN_LDM_CKPT=""

# ── Stage selector ───────────────────────────────────────────────────────────
# FULL_PIPELINE=1 → fresh run, checkpoints passed between stages via PIPELINE_*
# FULL_PIPELINE=0 → single stage, checkpoint paths come from PRETRAIN_* above
RUN_FDP=1; RUN_VAE=1; RUN_LDM=1; RUN_EVAL=1; FULL_PIPELINE=1
if [ $# -ge 1 ]; then
    RUN_FDP=0; RUN_VAE=0; RUN_LDM=0; RUN_EVAL=0; FULL_PIPELINE=0
    case "$1" in
        fdp)  RUN_FDP=1  ;;
        vae)  RUN_VAE=1  ;;
        ldm)  RUN_LDM=1  ;;
        eval) RUN_EVAL=1 ;;
        *) die "Unknown stage '$1'. Use: fdp | vae | ldm | eval" ;;
    esac
fi

# ── Run directory ─────────────────────────────────────────────────────────────
NOW=$(date '+%Y-%m-%dT%H-%M-%S')
if   [ "$FULL_PIPELINE" -eq 1 ]; then RUN_DIR="$LOGS/${NOW}_full_pipeline"
elif [ "$RUN_FDP"  -eq 1 ];      then RUN_DIR="$LOGS/${NOW}_fdp_pipeline"
elif [ "$RUN_VAE"  -eq 1 ];      then RUN_DIR="$LOGS/${NOW}_vae_pipeline"
elif [ "$RUN_LDM"  -eq 1 ];      then RUN_DIR="$LOGS/${NOW}_ldm_pipeline"
else                                   RUN_DIR="$LOGS/${NOW}_eval_pipeline"
fi
mkdir -p "$RUN_DIR/fdp" "$RUN_DIR/vae" "$RUN_DIR/ldm"
LOGFILE="$RUN_DIR/pipeline.log"

log_hyperparameters() {
    log "Hyperparameters:"
    log "  FDP: min_epochs=$FDP_MIN_EPOCHS  max_epochs=$FDP_MAX_EPOCHS  batch=$FDP_BATCH  val_every=$FDP_VAL_EVERY  patience=$FDP_EARLY_STOPPING_PATIENCE"
    log "  VAE: min_epochs=$VAE_MIN_EPOCHS  max_epochs=$VAE_MAX_EPOCHS  batch=$VAE_BATCH  val_every=$VAE_VAL_EVERY  patience=$VAE_EARLY_STOPPING_PATIENCE"
    log "  LDM: min_epochs=$LDM_MIN_EPOCHS  max_epochs=$LDM_MAX_EPOCHS  batch=$LDM_BATCH  val_every=$LDM_VAL_EVERY  patience=$LDM_EARLY_STOPPING_PATIENCE"
    log "  EVAL: max_samples=$VAL_MAX_SAMPLES"
    log "Runtime:"
    log "  PYTHON=$PYTHON"
    log "  ROOT=$ROOT"
    log "  RUN_DIR=$RUN_DIR"
    log "  GPU_LIST=$GPU_LIST"
    log "Pretrained checkpoints:"
    log "  PRETRAIN_FDP_CKPT=${PRETRAIN_FDP_CKPT:-<none>}"
    log "  PRETRAIN_VAE_CKPT=${PRETRAIN_VAE_CKPT:-<none>}"
    log "  PRETRAIN_LDM_CKPT=${PRETRAIN_LDM_CKPT:-<none>}"
}

# ── GPU setup ────────────────────────────────────────────────────────────────
# Specify GPUs via env var or second argument:
#   GPU_LIST=2,3 bash run_pipeline.sh [stage]
#   bash run_pipeline.sh [stage] 2,3
if [ $# -ge 2 ]; then
    GPU_LIST="$2"
else
    GPU_LIST="${GPU_LIST:-0,1,2,3}"
fi
NGPU=$(echo "$GPU_LIST" | tr ',' '\n' | grep -c .)
FIRST_GPU=$(echo "$GPU_LIST" | cut -d',' -f1)
# PyTorch Lightning sees remapped indices 0..N-1 via CUDA_VISIBLE_DEVICES
GPUS_ARG=$(python3 -c "print(','.join(str(i) for i in range($NGPU)) + ',')")
NGPU_TOTAL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
log "Found $NGPU_TOTAL GPUs total, using $NGPU: [$GPU_LIST]"

# ── Dynamic checkpoint finders ────────────────────────────────────────────────
# Use find so pipefail never fires on a missing glob.
_latest_fdp_ckpt() {
    find "$LOGS" -path "*/fdp/checkpoints/last.ckpt" \
        -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}
_latest_vae_ckpt() {
    find "$LOGS" -path "*/vae/checkpoints/last.ckpt" \
        -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}
_latest_ldm_ckpt() {
    find "$LOGS" -path "*/ldm/checkpoints/last.ckpt" \
        -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

# Inter-stage state: set by each stage during a full pipeline run
PIPELINE_FDP_CKPT=""
PIPELINE_VAE_CKPT=""

# =============================================================================
# Stage 1 — FDP (Frequency Reconstruction Module)
# =============================================================================
run_fdp() {
    log "============================================================"
    log " Stage 1: FDP training  (epochs=$FDP_MIN_EPOCHS–$FDP_MAX_EPOCHS, 4×GPU, batch=$FDP_BATCH/GPU, patience=$FDP_EARLY_STOPPING_PATIENCE)"
    log " Output  : $RUN_DIR/fdp/"
    log "============================================================"

    local no_resume_flag="" pretrain_arg=""
    if [ "$FULL_PIPELINE" -eq 1 ]; then
        no_resume_flag="--no-resume"
    elif [ -f "$PRETRAIN_FDP_CKPT" ]; then
        pretrain_arg="--pretrain $PRETRAIN_FDP_CKPT"
        log "Loading pretrained FDP weights from: $PRETRAIN_FDP_CKPT"
    fi

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$GPU_LIST \
    $PYTHON fdp/train.py \
        --config                   configs/fdp/fdp.yaml \
        --out_dir                  "$RUN_DIR/fdp" \
        --min_epochs               $FDP_MIN_EPOCHS \
        --epochs                   $FDP_MAX_EPOCHS \
        --batch_size               $FDP_BATCH \
        --num_workers              16 \
        --val_every                $FDP_VAL_EVERY \
        --early_stopping_patience  $FDP_EARLY_STOPPING_PATIENCE \
        --devices                  $NGPU \
        $no_resume_flag \
        $pretrain_arg \
        2>&1 | tee -a "$RUN_DIR/fdp/train.log"

    [ -f "$RUN_DIR/fdp/checkpoints/last.ckpt" ] \
        || die "fdp/checkpoints/last.ckpt not found after training"
    PIPELINE_FDP_CKPT=$(realpath "$RUN_DIR/fdp/checkpoints/last.ckpt")
    log "FDP done → $PIPELINE_FDP_CKPT"

    log "FDP: generating samples…"
    CUDA_VISIBLE_DEVICES=$FIRST_GPU $PYTHON scripts/sample_vis.py fdp \
        --ckpt "$RUN_DIR/fdp/checkpoints" \
        --out_dir "$RUN_DIR/fdp" \
        --n 6 2>&1 | tee -a "$RUN_DIR/fdp/train.log"
    log "FDP samples → $RUN_DIR/fdp/samples_fdp.png"
}

# =============================================================================
# Stage 2 — VAE (KL-regularised autoencoder)
# =============================================================================

_make_vae_cfg() {
    mkdir -p "$RUN_DIR/vae/configs"
    local out="$RUN_DIR/vae/configs/vae_pipeline.yaml"
    $PYTHON - <<PYEOF
import yaml
with open("configs/vae/VAE_kl_32x32x4.yaml") as f:
    c = yaml.safe_load(f)
c["data"]["params"]["batch_size"]  = $VAE_BATCH
c["data"]["params"]["num_workers"] = 16
c["lightning"]["trainer"]["min_epochs"]       = $VAE_MIN_EPOCHS
c["lightning"]["trainer"]["max_epochs"]       = $VAE_MAX_EPOCHS
c["lightning"]["trainer"]["strategy"]         = "ddp_find_unused_parameters_true"
c["lightning"]["trainer"]["check_val_every_n_epoch"] = $VAE_VAL_EVERY
c["lightning"]["trainer"]["log_every_n_steps"] = 50
c["lightning"]["trainer"]["enable_progress_bar"] = True
c["lightning"]["callbacks"]["early_stopping"] = {
    "target": "main.EarlyStoppingAfterMinEpochs",
    "params": {
        "monitor":   "val/rec_loss",
        "mode":      "min",
        "patience":  $VAE_EARLY_STOPPING_PATIENCE,
        "min_delta": 1e-4,
        "verbose":   False,
    },
}
with open("$out", "w") as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True)
print("wrote $out", file=__import__('sys').stderr)
PYEOF
    echo "$out"
}

run_vae() {
    log "============================================================"
    log " Stage 2: VAE training  (epochs=$VAE_MIN_EPOCHS–$VAE_MAX_EPOCHS, 4×GPU, batch=$VAE_BATCH/GPU, patience=$VAE_EARLY_STOPPING_PATIENCE)"
    log " Output  : $RUN_DIR/vae/"
    log "============================================================"

    if [ "$FULL_PIPELINE" -eq 0 ]; then
        [ -f "$PRETRAIN_FDP_CKPT" ] || die "PRETRAIN_FDP_CKPT not found: $PRETRAIN_FDP_CKPT"
        log "Single-stage VAE: FDP=$PRETRAIN_FDP_CKPT"
    fi

    local cfg
    cfg=$(_make_vae_cfg)
    log "Using config: $cfg"

    local resume_arg=""
    if [ "$FULL_PIPELINE" -eq 0 ] && [ -f "$PRETRAIN_VAE_CKPT" ]; then
        log "Loading pretrained VAE from: $PRETRAIN_VAE_CKPT"
        resume_arg="--resume $PRETRAIN_VAE_CKPT"
    fi

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$GPU_LIST \
    $PYTHON main.py \
        --base        "$cfg" \
        --logdir      "$RUN_DIR/vae" \
        --fixed-logdir \
        -t --gpus $GPUS_ARG \
        --scale_lr False \
        --no-test \
        $resume_arg \
        2>&1 | tee -a "$RUN_DIR/vae/train.log"

    local ckpt
    ckpt=$(_latest_vae_ckpt); [ -n "$ckpt" ] || die "No VAE checkpoint found in $RUN_DIR/vae/"
    PIPELINE_VAE_CKPT=$(realpath "$ckpt")
    log "VAE done → $PIPELINE_VAE_CKPT"

    log "VAE: generating samples…"
    CUDA_VISIBLE_DEVICES=$FIRST_GPU $PYTHON scripts/sample_vis.py vae \
        --ckpt "$RUN_DIR/vae/checkpoints" \
        --config "configs/vae/VAE_kl_32x32x4.yaml" \
        --out_dir "$RUN_DIR/vae" \
        --n 6 2>&1 | tee -a "$RUN_DIR/vae/train.log"
    log "VAE samples → $RUN_DIR/vae/samples_vae.png"
}

# =============================================================================
# Stage 3 — LDM (Latent Diffusion Model)
# =============================================================================

_make_ldm_cfg() {
    mkdir -p "$RUN_DIR/ldm/configs"
    local out="$RUN_DIR/ldm/configs/ldm_pipeline.yaml"

    # Full pipeline: use checkpoints passed from previous stages.
    # Single stage:  use the PRETRAIN_* paths hardcoded above.
    local fdp_ckpt vae_ckpt ldm_ckpt_arg="None"
    if [ "$FULL_PIPELINE" -eq 1 ]; then
        fdp_ckpt="$PIPELINE_FDP_CKPT"
        vae_ckpt="$PIPELINE_VAE_CKPT"
    else
        fdp_ckpt=$(realpath "$PRETRAIN_FDP_CKPT")
        vae_ckpt=$(realpath "$PRETRAIN_VAE_CKPT")
        [ -f "$PRETRAIN_LDM_CKPT" ] && ldm_ckpt_arg="\"$(realpath "$PRETRAIN_LDM_CKPT")\""
    fi
    log "LDM will load FDP  : $fdp_ckpt" >&2
    log "LDM will load VAE  : $vae_ckpt" >&2
    [ "$ldm_ckpt_arg" != "None" ] && log "LDM will load LDM  : $(realpath "$PRETRAIN_LDM_CKPT")" >&2

    $PYTHON - <<PYEOF
import yaml
src = "configs/ldm/ldm_cond_frq_0d1.yaml"
with open(src) as f:
    c = yaml.safe_load(f)
c["model"]["params"]["cond_stage_config"]["params"]["ckpt_path"]  = "$fdp_ckpt"
c["model"]["params"]["first_stage_config"]["params"]["ckpt_path"] = "$vae_ckpt"
c["model"]["params"]["ckpt_path"] = $ldm_ckpt_arg
c["model"]["params"]["saved_path"] = None
c["data"]["params"]["batch_size"]  = $LDM_BATCH
c["data"]["params"]["num_workers"] = 16
c["lightning"]["trainer"]["min_epochs"]          = $LDM_MIN_EPOCHS
c["lightning"]["trainer"]["max_epochs"]          = $LDM_MAX_EPOCHS
c["lightning"]["trainer"]["strategy"]            = "ddp_find_unused_parameters_true"
c["lightning"]["trainer"]["check_val_every_n_epoch"] = $LDM_VAL_EVERY
c["lightning"]["trainer"]["log_every_n_steps"]   = 50
c["lightning"]["trainer"]["enable_progress_bar"] = True
c["lightning"]["callbacks"]["early_stopping"] = {
    "target": "main.EarlyStoppingAfterMinEpochs",
    "params": {
        "monitor":   "val/loss_simple_ema",
        "mode":      "min",
        "patience":  $LDM_EARLY_STOPPING_PATIENCE,
        "min_delta": 1e-4,
        "verbose":   False,
    },
}
with open("$out", "w") as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True)
print("wrote $out", file=__import__('sys').stderr)
PYEOF
    echo "$out"
}

run_ldm() {
    log "============================================================"
    log " Stage 3: LDM training  (epochs=$LDM_MIN_EPOCHS–$LDM_MAX_EPOCHS, 4×GPU, batch=$LDM_BATCH/GPU, patience=$LDM_EARLY_STOPPING_PATIENCE)"
    log " Output  : $RUN_DIR/ldm/"
    log "============================================================"

    if [ "$FULL_PIPELINE" -eq 1 ]; then
        local _fc; _fc=$(_latest_fdp_ckpt); [ -n "$_fc" ] || die "No FDP checkpoint found – run FDP stage first"
        local _vc; _vc=$(_latest_vae_ckpt); [ -n "$_vc" ] || die "No VAE checkpoint found – run VAE stage first"
    else
        [ -f "$PRETRAIN_FDP_CKPT" ] || die "PRETRAIN_FDP_CKPT not found: $PRETRAIN_FDP_CKPT"
        [ -f "$PRETRAIN_VAE_CKPT" ] || die "PRETRAIN_VAE_CKPT not found: $PRETRAIN_VAE_CKPT"
        log "Single-stage LDM: FDP=$PRETRAIN_FDP_CKPT"
        log "Single-stage LDM: VAE=$PRETRAIN_VAE_CKPT"
    fi

    local cfg
    cfg=$(_make_ldm_cfg)
    log "Using config: $cfg"

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$GPU_LIST \
    $PYTHON main.py \
        --base        "$cfg" \
        --logdir      "$RUN_DIR/ldm" \
        --fixed-logdir \
        -t --gpus $GPUS_ARG \
        --scale_lr False \
        --no-test \
        2>&1 | tee -a "$RUN_DIR/ldm/train.log"

    local ckpt
    ckpt=$(_latest_ldm_ckpt); [ -n "$ckpt" ] || die "No LDM checkpoint found in $RUN_DIR/ldm/"
    PIPELINE_LDM_CKPT=$(realpath "$ckpt")
    log "LDM done → $PIPELINE_LDM_CKPT"

    log "LDM: generating samples…"
    CUDA_VISIBLE_DEVICES=$FIRST_GPU $PYTHON scripts/sample_vis.py ldm \
        --ckpt "$RUN_DIR/ldm/checkpoints" \
        --config "$cfg" \
        --out_dir "$RUN_DIR/ldm" \
        --n 6 2>&1 | tee -a "$RUN_DIR/ldm/train.log"
    log "LDM samples → $RUN_DIR/ldm/samples_ldm.png"
}

# =============================================================================
# Stage 3 — Evaluation (AUROC / AUPRC / Dice on BraTS test set)
# =============================================================================

_make_eval_cfg() {
    mkdir -p "$RUN_DIR/eval_results/configs"
    local out="$RUN_DIR/eval_results/configs/eval_pipeline.yaml"

    local fdp_ckpt vae_ckpt
    if [ "$FULL_PIPELINE" -eq 1 ]; then
        fdp_ckpt="$PIPELINE_FDP_CKPT"
        vae_ckpt="$PIPELINE_VAE_CKPT"
    else
        fdp_ckpt=$(realpath "$PRETRAIN_FDP_CKPT")
        vae_ckpt=$(realpath "$PRETRAIN_VAE_CKPT")
    fi
    local ldm_ckpt
    if [ "$FULL_PIPELINE" -eq 1 ]; then
        ldm_ckpt=$(_latest_ldm_ckpt); [ -n "$ldm_ckpt" ] || die "No LDM checkpoint found – run LDM stage first"
    elif [ -n "$PRETRAIN_LDM_CKPT" ]; then
        ldm_ckpt="$PRETRAIN_LDM_CKPT"
    else
        ldm_ckpt=$(_latest_ldm_ckpt); [ -n "$ldm_ckpt" ] || die "No LDM checkpoint found – set PRETRAIN_LDM_CKPT or run LDM stage first"
    fi
    ldm_ckpt=$(realpath "$ldm_ckpt")

    log "Eval will load FDP : $fdp_ckpt" >&2
    log "Eval will load VAE : $vae_ckpt" >&2
    log "Eval will load LDM : $ldm_ckpt" >&2

    $PYTHON - <<PYEOF
import yaml
src = "configs/eval/ldm_cond_frq_0d1.yaml"
with open(src) as f:
    c = yaml.safe_load(f)
c["model"]["params"]["ckpt_path"]                                        = "$ldm_ckpt"
c["model"]["params"]["cond_stage_config"]["params"]["ckpt_path"]         = "$fdp_ckpt"
c["model"]["params"]["first_stage_config"]["params"]["ckpt_path"]        = "$vae_ckpt"
c["model"]["params"]["saved_path"] = "$RUN_DIR/eval_results"
c["data"]["params"]["test"]["params"]["max_samples"] = $VAL_MAX_SAMPLES
with open("$out", "w") as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True)
print("wrote $out", file=__import__('sys').stderr)
PYEOF
    echo "$out"
}

run_eval() {
    log "============================================================"
    log " Stage 3: Evaluation  (BraTS test set, single GPU)"
    log " Output  : $RUN_DIR/eval_results/"
    log "============================================================"

    if [ "$FULL_PIPELINE" -eq 0 ]; then
        [ -f "$PRETRAIN_FDP_CKPT" ] || die "PRETRAIN_FDP_CKPT not found: $PRETRAIN_FDP_CKPT"
        [ -f "$PRETRAIN_VAE_CKPT" ] || die "PRETRAIN_VAE_CKPT not found: $PRETRAIN_VAE_CKPT"
        if [ -n "$PRETRAIN_LDM_CKPT" ]; then
            [ -f "$PRETRAIN_LDM_CKPT" ] || die "PRETRAIN_LDM_CKPT not found: $PRETRAIN_LDM_CKPT"
        else
            local _lc; _lc=$(_latest_ldm_ckpt); [ -n "$_lc" ] || die "No LDM checkpoint found – set PRETRAIN_LDM_CKPT or run LDM stage first"
        fi
    fi

    local cfg
    cfg=$(_make_eval_cfg)
    log "Using eval config: $cfg"

    CUDA_VISIBLE_DEVICES=$FIRST_GPU \
    $PYTHON main.py \
        --base "$cfg" \
        --logdir      "$RUN_DIR" \
        --fixed-logdir \
        --gpus 0, \
        --scale_lr False \
        2>&1 | tee -a "$RUN_DIR/eval.log"

    log "Evaluation done → $RUN_DIR/eval_results/eval_metrics.log"
}

# =============================================================================
# Main
# =============================================================================
log "Pipeline start  (run_dir=$RUN_DIR)"
log "Stages: FDP=$RUN_FDP  VAE=$RUN_VAE  LDM=$RUN_LDM  EVAL=$RUN_EVAL"
log_hyperparameters
START=$(date +%s)

[ "$RUN_FDP"  -eq 1 ] && run_fdp
[ "$RUN_VAE"  -eq 1 ] && run_vae
[ "$RUN_LDM"  -eq 1 ] && run_ldm
[ "$RUN_EVAL" -eq 1 ] && run_eval

ELAPSED=$(( $(date +%s) - START ))
log "Pipeline complete in $(( ELAPSED/3600 ))h $(( ELAPSED%3600/60 ))m $(( ELAPSED%60 ))s"
