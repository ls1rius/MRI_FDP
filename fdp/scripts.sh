# example of FDP+LDM

# VAE training
CUDA_VISIBLE_DEVICES=0 python main.py \
--base configs/vae/VAE_kl_32x32x4.yaml \
-t --gpus 0, --scale_lr False


# FDP Stage one training:
CUDA_VISIBLE_DEVICES=0 python main.py \
--base configs/stage1/FDP_ldm_cond_zmask_x0_prior_stage1_frq_0.1_remove_low_frequency_single_0.10.yaml \
-t --gpus 0, --scale_lr False

# FDP Stage two training:
CUDA_VISIBLE_DEVICES=0 python main.py \
--base configs/stage2/mri_ldm_vae_cond_zmask_x0_prior_stage2_frq_0.1_remove_low_frequency_single_0.10.yaml \
-t --gpus 0, --scale_lr False

# FDP test and visualization:
CUDA_VISIBLE_DEVICES=0 python main.py \
--base configs/stage2/FDP_ldm_cond_zmask_x0_prior_test_frq_0.1_remove_low_frequency_single_0.10.yaml \
--gpus 0
