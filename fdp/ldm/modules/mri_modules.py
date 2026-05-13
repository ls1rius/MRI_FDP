import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from ldm.modules.x_transformer import Attention
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import default
import scipy.stats as stats
import copy
from ldm.modules.diffusionmodules.model import Encoder, Decoder
import pywt
from skimage.transform import resize
from skimage import measure

class MRICrossFusion(nn.Module):
    def __init__(
        self,
        image_size=32,
        n_embed=4,
        n_layer=8,
        n_channels=4,
        n_prior_head=1,
        frq_mask_ratio=0.15
        ):
        super().__init__()
        self.image_size = image_size
        self.n_embed = n_embed
        self.n_channels = n_channels
        self.n_prior_head = n_prior_head
        # self.mri_prior_info = nn.Parameter(torch.load('./data/IXI/vae_embed_{}ac_non_kmpp_high.pth'.format(str(self.n_prior_head))))
        # self.mri_prior_info = nn.Parameter(torch.randn(self.n_prior_head*n_embed*2, image_size, image_size))
        self.attn = Attention(image_size*image_size*n_embed, n_layer)
        # self.attn = Attention(image_size*image_size, n_layer)
        # nn.init.normal_(self.mri_prior_info)
        # self.ffn_proj = nn.Sequential(nn.Linear(image_size*image_size*n_embed, image_size*image_size*n_embed//16), 
        #                                 nn.Linear(image_size*image_size*n_embed//16, image_size*image_size*n_embed))
        # self.conv_proj = nn.Sequential(
        #     nn.Conv2d(n_embed*2, n_embed*8, (3, 3), 1, 1),
        #     nn.PReLU(),
        #     nn.Conv2d(n_embed*8, n_embed, (1, 1), 1, 0),
        # )
        ###
        self.mri_prior_info_real = nn.Parameter(torch.load('./data/IXI/t2_hf_0d3_embed_{}ac_real_frq.pth'.format(str(self.n_prior_head))))
        self.bn_for_frq_in_real = nn.BatchNorm2d(1)
        self.attn_for_frq_real = Attention(int(256 * frq_mask_ratio // 2 * 2)*int(256 * frq_mask_ratio // 2 * 2) * 1, 8)
        self.bn_for_frq_out_real = nn.BatchNorm2d(1)
        self.conv_for_frq_real = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), 1, 1),
            nn.PReLU(),
            nn.Conv2d(16, 1, (1, 1), 1, 0),
        )
        self.mri_prior_info_imag = nn.Parameter(torch.load('./data/IXI/t2_hf_0d3_embed_{}ac_imag_frq.pth'.format(str(self.n_prior_head))))
        self.bn_for_frq_in_imag = nn.BatchNorm2d(1)
        self.attn_for_frq_imag = Attention(int(256 * frq_mask_ratio // 2 * 2)*int(256 * frq_mask_ratio // 2 * 2) * 1, 8)
        self.bn_for_frq_out_imag = nn.BatchNorm2d(1)
        self.conv_for_frq_imag = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), 1, 1),
            nn.PReLU(),
            nn.Conv2d(16, 1, (1, 1), 1, 0),
        )
        ###
        # self.attn_for_frq_real_donut = Attention(int(256 * 0.30 // 2 * 2)*int(256 * 0.30 // 2 * 2) * 1, 8)
        # self.conv_for_frq_real_donut = nn.Sequential(
        #     nn.Linear(4476, 2048),
        #     nn.Linear(2048, 2048),
        #     nn.PReLU(),
        #     nn.Linear(2048, 1300)
        #     # nn.Conv2d(1, 16, (3, 3), 1, 1),
        #     # # nn.PReLU(),
        #     # nn.Conv2d(16, 1, (1, 1), 1, 0),

        #     # nn.Conv2d(1, 16, (3, 3), 1, 1),
        #     # nn.Conv2d(16, 64, (3, 3), 1, 1),
        #     # nn.PReLU(),
        #     # nn.Conv2d(64, 16, (1, 1), 1, 0),
        #     # nn.Conv2d(16, 1, (1, 1), 1, 0),
        # )
        ###
        # # self.bn_for_frq_real_donut = nn.BatchNorm2d(1)
        # self.encode_for_frq_real_donut = Encoder(
        #     double_z=False, 
        #     z_channels=1, 
        #     resolution=256, 
        #     in_channels=1,
        #     out_ch=1,
        #     ch=128,
        #     ch_mult=[ 1, 2, 4, 4],
        #     num_res_blocks=2,
        #     attn_resolutions=[ ],
        #     dropout=0.0
        #   )
        # self.attn_for_frq_real_donut = Attention(32 * 32 * 1, 8)
        # self.conv_for_frq_real_donut = nn.Sequential(
        #     nn.Linear(32 * 32, 1024),
        #     nn.PReLU(),
        #     nn.Linear(1024, int(256 * 0.15 // 2 * 2)*int(256 * 0.15 // 2 * 2) * 1),
        # )
    def logkp(self, x, k=10000):
        return x
        # return torch.log(x + k)
        # return torch.tanh(x / 100)
        # return torch.log(1 + torch.abs(x)) * torch.sign(x) * (1.0/8)

    def rlogkp(self, x, k=10000):
        return x
        # return torch.exp(x) - k
        # return torch.arctanh(x) * 100.0
        # return torch.sign(x) * (torch.exp(torch.abs(x * (8.0))) - 1.0)

    def add_gauss_coarse_noise(self, img_tensor, need_norm=True, need_mfix=True):
        # img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        N, C, H, W = img_tensor.shape

        upsample_scale_ratio = 8
        theta2 = torch.sqrt(torch.tensor(0.2))
        noise_down = torch.randn((N, C, H//upsample_scale_ratio, W//upsample_scale_ratio)).to(img_tensor.device) * theta2
        # mask_down = torch.ones_like(noise_down)
        # mask_down = torch.where(torch.randn_like(noise_down) < 0.01, 1.0, 0.0)
        
        noise_down_up = nn.Upsample(scale_factor=upsample_scale_ratio, mode='bilinear', align_corners=True)(noise_down)
        # mask_down_up  = nn.Upsample(scale_factor=upsample_scale_ratio, mode='bilinear', align_corners=True)(mask_down)

        # noise_down_up *= mask_down_up.round()
        img_back_mag = img_tensor + noise_down_up

        img_back_mag = torch.where(img_tensor > -1, img_back_mag, -1)
        # img_back_mag = torch.where(img_tensor > 0, img_back_mag, 0) if (need_norm and need_mfix) else img_back_mag
        # img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        # img_back_mag = torch.where(img_tensor > 0, img_back_mag, -1) if (need_norm and need_mfix) else img_back_mag
        return img_back_mag


    def remove_noise_region_slice(self, x):
        x_cur = copy.deepcopy(x).cpu()
        x_cur_mean = x_cur[x_cur>0].mean()
        cc_x = measure.label(x_cur > 4 * x_cur_mean, connectivity=2)
        props = measure.regionprops(cc_x)
        areas = [region.area for region in props]
        for prop in props:
            if prop['filled_area'] <= 150:
                x_cur[cc_x == prop['label']] = 0
        return x_cur

    def remove_noise_region(self, x, need_norm=True):
        x = (self.mri_norm(x) + 1.0) / 2.0 if need_norm else x
        N, C, H, W  = x.shape
        x_list_nc = []
        for i in range(N):
            x_list_c = []
            for j in range(C):
                x_list_c.append(self.remove_noise_region_slice(x[i,j]))
            x_list_nc.append(torch.stack(x_list_c, dim=0))
        x_nc = torch.stack(x_list_nc, dim=0).to(x.device)
        x_nc = self.mri_norm(x_nc) if need_norm else x_nc
        return x_nc

    def get_combine_img_low_remove_noise_region(self, img_tensor, img_fft_shift_part, mask_ratio=None, need_norm=True):
        mask_ratio = default(mask_ratio, np.random.uniform(0, 0.1))
        low_frq_img = self.remove_low_frequency_single(img_tensor, mask_ratio, need_norm)
        low_frq_img_wo_noise = self.remove_noise_region(low_frq_img, need_norm)
        low_frq_img_wo_noise_combined = self.get_combine_img_low(low_frq_img_wo_noise, img_fft_shift_part, mask_ratio, need_norm)
        return low_frq_img_wo_noise_combined

    def get_pywt(self, x, need_norm=True):
        x = (self.mri_norm(x) + 1.0) / 2.0 if need_norm else x
        N, C, H, W = x.shape
        # wavelet
        LH_list = []
        HL_list = []
        HH_list = []
        for ind in range(len(x)):
            LL, (LH, HL, HH) = pywt.dwt2(x[ind, 0].cpu(), 'haar')
            LH_list.append(torch.Tensor(np.array([resize(LH, (H, W))])).to(x.device))
            HL_list.append(torch.Tensor(np.array([resize(HL, (H, W))])).to(x.device))
            HH_list.append(torch.Tensor(np.array([resize(HH, (H, W))])).to(x.device))
        LH_all = torch.stack(LH_list, dim=0)
        HL_all = torch.stack(HL_list, dim=0)
        HH_all = torch.stack(HH_list, dim=0)
        LH_all = self.mri_norm(LH_all) if need_norm else LH_all
        HL_all = self.mri_norm(HL_all) if need_norm else HL_all
        HH_all = self.mri_norm(HH_all) if need_norm else HH_all
        return LH_all, HL_all, HH_all

    def get_mri_prior_info_frq(self, x, mask_ratio, mode='real'):
        N, C, H, W = x.shape
        H_hf, W_hf = int(256 * 0.3), int(256 * 0.3)
        # H_hf, W_hf = int(256 * 1.0), int(256 * 1.0)
        cond = getattr(self, 'mri_prior_info_{}'.format(mode))
        cond = cond.reshape(1, self.n_prior_head, H_hf, W_hf)
        cond = cond.repeat((N, 1, 1, 1))
        ### crop
        rows, cols = 256, 256
        crow, ccol = H_hf // 2, W_hf // 2
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        cond = cond[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2]
        ### 
        x = self.logkp(x)
        cond = self.logkp(cond)
        x = getattr(self, 'bn_for_frq_in_{}'.format(mode))(x)
        cond = getattr(self, 'bn_for_frq_in_{}'.format(mode))(cond.reshape(N*self.n_prior_head, -1, H, W))
        cond_mt = cond.reshape(N, self.n_prior_head, -1, H, W)
        data = getattr(self, 'attn_for_frq_{}'.format(mode))(x.flatten(start_dim=1).unsqueeze(dim=1), cond_mt.flatten(start_dim=2))[0]
        data = data.reshape(N, C, H, W)
        # data = getattr(self, 'bn_for_frq_out_{}'.format(mode))(data)
        data = getattr(self, 'conv_for_frq_{}'.format(mode))(data)
        data = self.rlogkp(data)
        return data

    def get_mri_prior_info_base(self, x):
        N = len(x)
        data = self.mri_prior_info
        data = data.unsqueeze(0)
        data = data.repeat((N, 1, 1, 1))
        return data

    def get_mri_prior_info_mode(self, x):
        data = self.get_mri_prior_info_base(x)
        data = DiagonalGaussianDistribution(data).mode()
        return data
    
    def get_mri_prior_info_sample(self, x):
        data = self.get_mri_prior_info_base(x)
        data = DiagonalGaussianDistribution(data).sample()
        return data

    # def get_mri_prior_info(self, x):
    #     return self.get_mri_prior_info_mode(x)

    def get_mri_prior_info_kmeans(self, x):
        N = len(x)
        data = self.mri_prior_info.reshape(self.n_prior_head*self.n_embed, self.image_size, self.image_size)
        data = data.unsqueeze(0)
        data = data.repeat((N, 1, 1, 1))
        return data

    def get_mri_prior_info_mode(self, x):
        data = self.get_mri_prior_info_kmeans(x)
        return data

    def simple_attn(self, q, k):
        v = k
        dots = torch.einsum('b i d, b j d -> b i j', torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(k, dim=-1))
        attn_w = F.softmax(dots, dim=-1)
        out_qkv = torch.einsum('b i j, b j d -> b i d', attn_w, v)
        out_ffn = self.ffn_proj(out_qkv)
        return out_ffn

    def get_mri_prior_info(self, x):
        N, C, H, W = x.shape
        cond = self.get_mri_prior_info_mode(x)
        cond_mt = cond.reshape(N, self.n_prior_head, -1, H, W)
        # data = self.simple_attn(x.flatten(start_dim=1).unsqueeze(dim=1), cond_mt.flatten(start_dim=2))
        data = self.attn(x.flatten(start_dim=1).unsqueeze(dim=1), cond_mt.flatten(start_dim=2))[0]
        # cond = self.get_mri_prior_info_mode_multi(x)
        # data = self.attn(cond.flatten(start_dim=1).unsqueeze(dim=1), x.flatten(start_dim=1).unsqueeze(dim=1))[0]
        # data = self.attn(x.flatten(start_dim=2), cond.flatten(start_dim=2))[0]
        # data = cond
        # data = x
        data = data.reshape(N, C, H, W)
        return data

    def get_mri_prior_info_mode_multi(self, x, hard=True):
        cond_multi = self.get_mri_prior_info_mode(x)
        if self.n_prior_head==1:
            return cond_multi
        cond = self.get_cond_max_cos_sim(cond_multi, x, hard=hard)
        return cond

    def get_ori_hfrq_cos_sim(self, ori, hfrq):
        ori_flatten = ori.flatten(start_dim=1).detach()
        hfrq_flatten = hfrq.flatten(start_dim=1).detach()
        ori_hfrq_cos_sim = torch.nn.functional.cosine_similarity(ori_flatten, hfrq_flatten, dim=-1)
        ori_hfrq_cos_sim = torch.clamp(ori_hfrq_cos_sim, min=0.0, max=1.0)
        return ori_hfrq_cos_sim

    def get_dftcond_cos_sim(self, x):
        cond = self.get_mri_prior_info_mode(x)
        cond_x_cos_sim =  self.get_cond_cos_sim(cond, x)
        cond_x_cos_sim_gumbel = self.gumbel_softmax(cond_x_cos_sim, hard=False)
        return cond_x_cos_sim_gumbel.max(dim=-1)[0]

    def get_cond_cos_sim(self, cond, x):
        N, CNH, H, W = cond.shape
        cond_mt = cond.reshape(N, self.n_prior_head, -1, H, W)
        # cond_flatten = cond_mt.flatten(start_dim=2)
        cond_flatten = cond_mt.flatten(start_dim=2).detach()
        x_flatten = x.flatten(start_dim=1).unsqueeze(dim=-2).detach()
        cond_x_cos_sim = torch.nn.functional.cosine_similarity(cond_flatten, x_flatten, dim=-1)
        return cond_x_cos_sim

    def get_cond_max_cos_sim(self, cond, x, hard=True):
        N, CNH, H, W = cond.shape
        cond_mt = cond.reshape(N, self.n_prior_head, -1, H, W)
        cond_x_cos_sim = self.get_cond_cos_sim(cond, x)
        # use gumbel
        cond_x_cos_sim_gumbel = self.gumbel_softmax(cond_x_cos_sim, hard=hard).reshape(N, self.n_prior_head, 1, 1, 1)
        # cond_x_cos_sim_gumbel = (self.gumbel_softmax(cond_x_cos_sim, hard=hard) * F.softmax(cond_x_cos_sim / 0.1, dim=-1)).reshape(N, self.n_prior_head, 1, 1, 1)
        cond_max_cos_sim = (cond_mt * cond_x_cos_sim_gumbel).sum(dim=1)
        return cond_max_cos_sim
        # cond_x_max_inds = cond_x_cos_sim.max(dim=-1)[1].detach().cpu()
        # cond_max_cos_sim = cond_mt[torch.arange(N), cond_x_max_inds]
        # return cond_max_cos_sim

    def enhance_frq_result_compare(self, x, y, mask_ratio=None, need_norm=True):
        mask_ratio = default(mask_ratio, 0.05)
        x = (self.mri_norm(x) + 1.0) / 2.0 if need_norm else x
        y = (self.mri_norm(y) + 1.0) / 2.0 if need_norm else y
        # Get image size
        N, C, rows, cols = x.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        # mask for low frq
        mask_low = torch.zeros((N, C, rows, cols), dtype=x.dtype, device=x.device)
        mask_low[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = 1
        mask_low = mask_low.bool()
        mask_high = ~mask_low
        # Transform image to frequency domain
        img_fft_x = fft.fftn(x, dim=(-2, -1))
        img_fft_shift_x = fft.fftshift(img_fft_x, dim=(-2, -1))
        img_fft_y = fft.fftn(y, dim=(-2, -1))
        img_fft_shift_y = fft.fftshift(img_fft_y, dim=(-2, -1))
        # Apply mask in low frequency domain
        x_low_norm_w = torch.linalg.norm(img_fft_shift_x.reshape(N, -1)[mask_low.reshape(N, -1)].real.reshape(N, -1), dim=-1)
        y_low_norm_w = torch.linalg.norm(img_fft_shift_y.reshape(N, -1)[mask_low.reshape(N, -1)].real.reshape(N, -1), dim=-1)
        xy_low_norm_scale = x_low_norm_w / y_low_norm_w
        # Apply mask in high frequency domain
        x_high_norm_w = torch.linalg.norm(img_fft_shift_x.reshape(N, -1)[mask_high.reshape(N, -1)].real.reshape(N, -1), dim=-1)
        y_high_norm_w = torch.linalg.norm(img_fft_shift_y.reshape(N, -1)[mask_high.reshape(N, -1)].real.reshape(N, -1), dim=-1)
        xy_high_norm_scale = x_high_norm_w / y_high_norm_w

        enhanced_data = self.enhance_frq_result(y, mask_ratio, need_norm=need_norm, low_scale=xy_low_norm_scale, high_scale=xy_high_norm_scale)
        return enhanced_data

    def enhance_frq_result(self, img_tensor, mask_ratio=None, need_norm=True, low_scale=0.9, high_scale=1.5):
        mask_ratio = default(mask_ratio, 0.05)
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        # Create mask
        mask = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device) * high_scale.reshape(N, 1, 1, 1)
        mask[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = 1 * low_scale.reshape(N, 1, 1, 1)
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift.real *= mask
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        # Compute magnitude (real part only)
        # img_back_mag = img_back.real
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag

    def get_combine_img_low(self, img_tensor, img_fft_shift_part, mask_ratio, need_norm=True, need_mfix=True):
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = img_fft_shift_part
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        img_back_mag = torch.abs(img_back)
        img_back_mag = torch.where(img_tensor > 0, img_back_mag, 0) if (need_norm and need_mfix) else img_back_mag
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag

    def fusion_combine_img_middle_whole(self, img_tensor, img_fft_shift_part, mask_ratio_low=0.05, mask_ratio_high=0.15, mask_ratio_end=1.00, need_norm=True):
        assert mask_ratio_end >= mask_ratio_high
        assert mask_ratio_high >= mask_ratio_low
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows_low = int(rows * mask_ratio_low)
        mask_cols_low = int(cols * mask_ratio_low)
        mask_rows_high = int(rows * mask_ratio_high)
        mask_cols_high = int(cols * mask_ratio_high)
        mask_rows_end = int(rows * mask_ratio_end)
        mask_cols_end = int(cols * mask_ratio_end)

        mask_high = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask_high[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0.0
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # get img fft shift real
        img_fft_shift_real = copy.deepcopy(img_fft_shift.real)
        img_fft_shift_real *= mask_high
        img_fft_shift_real[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = img_fft_shift_part.real
        img_fft_shift_real_donut = self.encode_for_frq_real_donut(self.logkp(img_fft_shift_real))
        img_fft_shift_real_donut_pred = self.conv_for_frq_real_donut(self.attn_for_frq_real_donut(img_fft_shift_real_donut.flatten(start_dim=1).unsqueeze(dim=1), img_fft_shift_real_donut.flatten(start_dim=1).unsqueeze(dim=1))[0].reshape(N, -1)).reshape(N, C, mask_rows_high, mask_cols_high)
        img_fft_shift_real_donut_pred = self.rlogkp(img_fft_shift_real_donut_pred)
        # calc part donut mask
        mask_all_donut = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask_all_donut[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0.0
        mask_all_donut[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = 1.0
        mask_part_donut = mask_all_donut
        ### crop middle
        img_fft_shift_real_donut_part_gdth = copy.deepcopy(img_fft_shift.real[~mask_part_donut.bool()].reshape(N, -1))
        img_fft_shift_real_donut_part_pred = img_fft_shift_real_donut_pred[(~mask_part_donut.bool())[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2]].reshape(N, -1)
        ### paste new low, middle into origin
        img_fft_shift.real[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = img_fft_shift_real_donut_pred
        img_fft_shift.real[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] += img_fft_shift_part.real
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag, img_fft_shift_real_donut_part_pred, img_fft_shift_real_donut_part_gdth

    def fusion_combine_img_middle(self, img_tensor, img_fft_shift_part, mask_ratio_low=0.05, mask_ratio_high=0.15, mask_ratio_end=0.30, need_norm=True):
        assert mask_ratio_end >= mask_ratio_high
        assert mask_ratio_high >= mask_ratio_low
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows_low = int(rows * mask_ratio_low // 2 * 2)
        mask_cols_low = int(cols * mask_ratio_low // 2 * 2)
        mask_rows_high = int(rows * mask_ratio_high // 2 * 2)
        mask_cols_high = int(cols * mask_ratio_high // 2 * 2)
        mask_rows_end = int(rows * mask_ratio_end // 2 * 2)
        mask_cols_end = int(cols * mask_ratio_end // 2 * 2)

        mask_ehl = torch.zeros((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask_ehl[:, :, crow-mask_rows_end//2:crow+mask_rows_end//2, ccol-mask_cols_end//2:ccol+mask_cols_end//2] = 1.0
        mask_ehl[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0.0
        mask_ehl[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = 1.0
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # get img fft shift real
        img_fft_shift_real = copy.deepcopy(img_fft_shift.real)
        img_fft_shift_real_donut = img_fft_shift_real[mask_ehl.bool()].reshape(N, -1)
        img_fft_shift_real_donut = self.logkp(img_fft_shift_real_donut)
        img_fft_shift_real_donut_pred = self.conv_for_frq_real_donut(img_fft_shift_real_donut)
        img_fft_shift_real_donut_pred = self.rlogkp(img_fft_shift_real_donut_pred)
        # calc part donut mask
        mask_all_donut = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask_all_donut[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0.0
        mask_all_donut[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = 1.0
        mask_part_donut = mask_all_donut[:, :, crow-mask_rows_end//2:crow+mask_rows_end//2, ccol-mask_cols_end//2:ccol+mask_cols_end//2]
        ### crop middle
        img_fft_shift_real_donut_part_gdth = copy.deepcopy(img_fft_shift.real[:, :, crow-mask_rows_end//2:crow+mask_rows_end//2, ccol-mask_cols_end//2:ccol+mask_cols_end//2]) * (~mask_part_donut.bool())
        
        img_fft_shift_real_donut_part_gdth_crop = img_fft_shift_real_donut_part_gdth[(~mask_part_donut.bool())].reshape(N, -1)
        img_fft_shift_real_donut_part_pred_crop = img_fft_shift_real_donut_pred
        # calc norm scale
        c0_norm_w = torch.linalg.norm(img_fft_shift_real_donut_part_gdth_crop, dim=-1)
        c1_norm_w = torch.linalg.norm(img_fft_shift_real_donut_part_pred_crop, dim=-1)
        c01_scale = (c0_norm_w / c1_norm_w).reshape(N, 1, 1, 1)
        ###
        mask_middle = torch.zeros((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask_middle[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 1.0
        mask_middle[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = 0.0
        img_fft_shift_tmp = torch.zeros_like(mask_middle).reshape(N, -1)
        img_fft_shift_tmp[mask_middle.bool().reshape(N, -1)] = img_fft_shift_real_donut_part_pred_crop.reshape(-1)
        img_fft_shift_tmp = img_fft_shift_tmp.reshape(mask_ehl.shape)
        ### paste new low, middle into origin
        img_fft_shift.real[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0
        img_fft_shift.real += img_fft_shift_tmp * c01_scale
        img_fft_shift.real[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] += img_fft_shift_part.real

        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag, img_fft_shift_real_donut_part_pred_crop, img_fft_shift_real_donut_part_gdth_crop

    # def fusion_combine_img_middle(self, img_tensor, img_fft_shift_part, mask_ratio_low=0.05, mask_ratio_high=0.15, mask_ratio_end=0.30, need_norm=True):
    #     assert mask_ratio_end >= mask_ratio_high
    #     assert mask_ratio_high >= mask_ratio_low
    #     img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
    #     # Get image size
    #     N, C, rows, cols = img_tensor.shape
    #     crow, ccol = rows // 2, cols // 2
    #     # Calculate size of the area to mask low-frequency information
    #     mask_rows_low = int(rows * mask_ratio_low // 2 * 2)
    #     mask_cols_low = int(cols * mask_ratio_low // 2 * 2)
    #     mask_rows_high = int(rows * mask_ratio_high // 2 * 2)
    #     mask_cols_high = int(cols * mask_ratio_high // 2 * 2)
    #     mask_rows_end = int(rows * mask_ratio_end // 2 * 2)
    #     mask_cols_end = int(cols * mask_ratio_end // 2 * 2)

    #     mask_high = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
    #     mask_high[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0.0
    #     # Transform image to frequency domain
    #     img_fft = fft.fftn(img_tensor, dim=(-2, -1))
    #     img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
    #     # get img fft shift real
    #     img_fft_shift_real = copy.deepcopy(img_fft_shift.real)
    #     img_fft_shift_real *= mask_high
    #     img_fft_shift_real[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = img_fft_shift_part.real
    #     img_fft_shift_real_donut = img_fft_shift_real[:, :, crow-mask_rows_end//2:crow+mask_rows_end//2, ccol-mask_cols_end//2:ccol+mask_cols_end//2]
    #     img_fft_shift_real_donut = self.logkp(img_fft_shift_real_donut)
    #     img_fft_shift_real_donut_pred = self.conv_for_frq_real_donut(self.attn_for_frq_real_donut(img_fft_shift_real_donut.flatten(start_dim=1).unsqueeze(dim=1), img_fft_shift_real_donut.flatten(start_dim=1).unsqueeze(dim=1))[0].reshape(N, 1, mask_rows_end, mask_cols_end))
    #     img_fft_shift_real_donut_pred = self.rlogkp(img_fft_shift_real_donut_pred)
    #     # calc part donut mask
    #     mask_all_donut = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
    #     mask_all_donut[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0.0
    #     mask_all_donut[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = 1.0
    #     mask_part_donut = mask_all_donut[:, :, crow-mask_rows_end//2:crow+mask_rows_end//2, ccol-mask_cols_end//2:ccol+mask_cols_end//2]
    #     ### crop middle
    #     img_fft_shift_real_donut_part_gdth = copy.deepcopy(img_fft_shift.real[:, :, crow-mask_rows_end//2:crow+mask_rows_end//2, ccol-mask_cols_end//2:ccol+mask_cols_end//2]) * (~mask_part_donut.bool())
    #     img_fft_shift_real_donut_part_pred = img_fft_shift_real_donut_pred * (~mask_part_donut.bool())

    #     img_fft_shift_real_donut_part_gdth_crop = img_fft_shift_real_donut_part_gdth[(~mask_part_donut.bool())].reshape(N, -1)
    #     img_fft_shift_real_donut_part_pred_crop = img_fft_shift_real_donut_part_pred[(~mask_part_donut.bool())].reshape(N, -1)
    #     # calc norm scale
    #     c0_norm_w = torch.linalg.norm(img_fft_shift_real_donut_part_gdth_crop, dim=-1)
    #     c1_norm_w = torch.linalg.norm(img_fft_shift_real_donut_part_pred_crop, dim=-1)
    #     c01_scale = (c0_norm_w / c1_norm_w).reshape(N, 1, 1, 1)
    #     #
    #     ### paste new low, middle into origin
    #     img_fft_shift.real[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0
    #     img_fft_shift.real[:, :, crow-mask_rows_end//2:crow+mask_rows_end//2, ccol-mask_cols_end//2:ccol+mask_cols_end//2] += img_fft_shift_real_donut_part_pred * c01_scale
    #     img_fft_shift.real[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] += img_fft_shift_part.real

    #     # Inverse transform to spatial domain
    #     img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
    #     img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
    #     img_back_mag = torch.abs(img_back)
    #     img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
    #     return img_back_mag, img_fft_shift_real_donut_part_pred_crop, img_fft_shift_real_donut_part_gdth_crop


    def get_combine_img_middle(self, img_tensor, img_fft_shift_part, mask_ratio_low=0.05, mask_ratio_high=0.15, need_norm=True):
        assert mask_ratio_high >= mask_ratio_low
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows_low = int(rows * mask_ratio_low)
        mask_cols_low = int(cols * mask_ratio_low)
        mask_rows_high = int(rows * mask_ratio_high)
        mask_cols_high = int(cols * mask_ratio_high)
        mask_high = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask_high[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0.20
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift.real *= mask_high
        img_fft_shift[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = img_fft_shift_part
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag


    def get_low_frequency_single(self, img_tensor, mask_ratio=None, need_norm=True, return_part=True):
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        if mask_ratio is None:
            return img_fft_shift
        if return_part:
            return img_fft_shift[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2]
        
        mask = torch.zeros((N, 1, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = 1
        return img_fft_shift * mask

    def get_low_time_single(self, img_tensor, img_fft_shift_part, mask_ratio=None, need_norm=True):
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)

        # Create mask
        img_fft_shift_real = torch.zeros((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        img_fft_shift_real[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = img_fft_shift_part.real

        img_fft_shift_imag = torch.zeros((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        img_fft_shift_imag[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = img_fft_shift_part.imag

        img_fft_shift = torch.complex(img_fft_shift_real, img_fft_shift_imag)
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        # Compute magnitude (real part only)
        # img_back_mag = img_back.real
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag


    def remove_low_frequency_add_noise(self, img_tensor, mask_ratio=None, need_norm=True):
        mask_ratio = default(mask_ratio, np.random.uniform(0, 0.1))
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, _, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        # if mask_rows<=4 or mask_cols<=4:
        #     return img_tensor
        # Create mask
        mask = torch.ones((N, 1, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = 0
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # add noise  by L10
        # kde_real = stats.gaussian_kde((img_fft_shift.real * (~mask.bool()).int())[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2].reshape(-1).cpu())
        # noise_real = kde_real.resample(N*1*(mask_rows//2*2)*(mask_cols//2*2)).reshape((N, 1, mask_cols//2*2, mask_cols//2*2))
        # kde_imag = stats.gaussian_kde((img_fft_shift.imag * (~mask.bool()).int())[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2].reshape(-1).cpu())
        # noise_imag = kde_imag.resample(N*1*(mask_rows//2*2)*(mask_cols//2*2)).reshape((N, 1, mask_cols//2*2, mask_cols//2*2))
        # noise = torch.complex(torch.Tensor(noise_real), torch.Tensor(noise_imag))
        noise_real = torch.randn_like(img_fft_shift.real)
        noise_imag = torch.randn_like(img_fft_shift.imag)
        noise = torch.complex(noise_real, noise_imag) * img_fft_shift * (~mask.bool()).int()
        # noise = noise.mean(dim=0).repeat(N, 1, 1, 1)
        # Apply mask in frequency domain
        # img_fft_shift[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = noise.to(img_fft_shift.device)
        img_fft_shift *= mask
        img_fft_shift += noise
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        # Compute magnitude (real part only)
        # img_back_mag = img_back.real
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag

    def remove_low_frequency(self, img_tensor, mask_ratio=None, need_norm=True):
        if mask_ratio is None or isinstance(mask_ratio, float):
            mask_ratio = default(mask_ratio, np.random.uniform(0, 0.1))
            return self.remove_low_frequency_single(img_tensor, mask_ratio, need_norm)
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, _, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Initialize mask with ones
        mask = torch.ones((N, 1, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        # Create mask for each image in the batch
        for i in range(N):
            mask_rows = int(rows * mask_ratio[i] // 2 * 2)
            mask_cols = int(cols * mask_ratio[i] // 2 * 2)
            mask[i, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = 0
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift *= mask
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        # Compute magnitude (real part only)
        # img_back_mag = img_back.real
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag


    def get_middle_frequency_single(self, img_tensor, mask_ratio_low=0.05, mask_ratio_high=0.15, need_norm=True):
        assert mask_ratio_high >= mask_ratio_low
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        N, _, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        mask_rows_low = int(rows * mask_ratio_low)
        mask_cols_low = int(cols * mask_ratio_low)
        mask_rows_high = int(rows * mask_ratio_high)
        mask_cols_high = int(cols * mask_ratio_high)
        mask = torch.ones((N, 1, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0
        mask[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = 1
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift *= mask
        return img_fft_shift


    def remove_middle_frequency_single(self, img_tensor, mask_ratio_low=0.05, mask_ratio_high=0.15, need_norm=True):
        assert mask_ratio_high >= mask_ratio_low
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        N, _, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        mask_rows_low = int(rows * mask_ratio_low)
        mask_cols_low = int(cols * mask_ratio_low)
        mask_rows_high = int(rows * mask_ratio_high)
        mask_cols_high = int(cols * mask_ratio_high)
        mask = torch.ones((N, 1, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask[:, :, crow-mask_rows_high//2:crow+mask_rows_high//2, ccol-mask_cols_high//2:ccol+mask_cols_high//2] = 0
        mask[:, :, crow-mask_rows_low//2:crow+mask_rows_low//2, ccol-mask_cols_low//2:ccol+mask_cols_low//2] = 1
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift *= mask
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        # Compute magnitude (real part only)
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag


    def remove_high_frequency_single(self, img_tensor, mask_ratio=None, need_norm=True):
        mask_ratio = default(mask_ratio, np.random.uniform(0, 0.1))
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        # Create mask
        mask = torch.zeros((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = 1
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift *= mask
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        # Compute magnitude (real part only)
        # img_back_mag = img_back.real
        img_back_mag = torch.abs(img_back)
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag

    def remove_low_frequency_single(self, img_tensor, mask_ratio=None, need_norm=True, need_mfix=True):
        mask_ratio = default(mask_ratio, np.random.uniform(0, 0.1))
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 if need_norm else img_tensor
        # Get image size
        N, C, rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2
        # Calculate size of the area to mask low-frequency information
        mask_rows = int(rows * mask_ratio // 2 * 2)
        mask_cols = int(cols * mask_ratio // 2 * 2)
        # Create mask
        mask = torch.ones((N, C, rows, cols), dtype=img_tensor.dtype, device=img_tensor.device)
        mask[:, :, crow-mask_rows//2:crow+mask_rows//2, ccol-mask_cols//2:ccol+mask_cols//2] = 0
        # Transform image to frequency domain
        img_fft = fft.fftn(img_tensor, dim=(-2, -1))
        img_fft_shift = fft.fftshift(img_fft, dim=(-2, -1))
        # Apply mask in frequency domain
        img_fft_shift *= mask
        # Inverse transform to spatial domain
        img_back_shift = fft.ifftshift(img_fft_shift, dim=(-2, -1))
        img_back = fft.ifftn(img_back_shift, dim=(-2, -1))
        # Compute magnitude (real part only)
        # img_back_mag = img_back.real
        img_back_mag = torch.abs(img_back)
        img_back_mag = torch.where(img_tensor > 0, img_back_mag, 0) if (need_norm and need_mfix) else img_back_mag
        img_back_mag = self.mri_norm(img_back_mag) if need_norm else img_back_mag
        return img_back_mag

    def canny_edge_detection(self, img_tensor, low_threshold=100, high_threshold=200):
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0 * 255
        # Ensure the tensor is on the CPU and convert it to NumPy array
        img_np = img_tensor.cpu().numpy()
        # Convert the tensor to a suitable format for OpenCV
        img_np = img_np.astype(np.uint8)
        img_np = img_np.squeeze(1)  # Remove the channel dimension if it exists
        # Apply Canny edge detection
        edges = []
        for img in img_np:
            edge = cv2.Canny(img, low_threshold, high_threshold)
            edges.append(edge)
        # Convert the edges list back to a tensor
        edges_np = np.stack(edges)
        edges_tensor = torch.from_numpy(edges_np).unsqueeze(1).to(img_tensor.device)  # Add the channel dimension back
        return self.mri_norm(edges_tensor)

    def edge_detection(self, img_tensor):
        img_tensor = (self.mri_norm(img_tensor) + 1.0) / 2.0
        # Sobel kernels for edge detection
        sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img_tensor.device).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=img_tensor.device).view(1, 1, 3, 3)
        # Apply Sobel kernels to the input tensor
        grad_x = F.conv2d(img_tensor, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(img_tensor, sobel_kernel_y, padding=1)
        # Compute gradient magnitude
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return self.mri_norm(edge_magnitude)

    def mri_norm(self, img_tensor):
        img_tensor = (img_tensor - img_tensor.amin(dim=(1, 2, 3), keepdim=True)) / (img_tensor.amax(dim=(1, 2, 3), keepdim=True) - img_tensor.amin(dim=(1, 2, 3), keepdim=True) + 1e-8)
        img_tensor = img_tensor * 2 - 1.0
        return img_tensor

    def gumbel_softmax(self, logits, tau=0.1, hard=True, eps=1e-8):
        y = logits
        # # generate Gumbel noise
        # noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        # # 计算Gumbel-Softmax分布
        # y = logits + noise
        y = F.softmax(y / tau, dim=-1)
        if hard:
            # 硬采样：选择概率最大的类别
            k = logits.size(-1)
            _, y_hard = y.max(dim=-1)
            y_hard = F.one_hot(y_hard, num_classes=k).float()
            y = (y_hard - y).detach() + y
        return y