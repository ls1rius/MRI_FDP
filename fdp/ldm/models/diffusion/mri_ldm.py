import os
import nibabel as nib
import torch
import torch.nn as nn
import scipy
import cv2
import numpy as np
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
from diffusers.models import AutoencoderKL
from .ddpm import LatentDiffusion
from ldm.util import default
from ldm.modules.metrics.test_utils import test_metrics, arverage_metric
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import random
# import pywt

class MRILatentDiffusion(LatentDiffusion):
    def __init__(self,
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
                 mask_strategy='None',
                 mask_stage='None',
                 m_ratio = 0.5,
                 max_rectangles = 6,
                 use_prior = False,
                 slide_test = False,
                 frq_mask_ratio = 0.05,
                 rlfs_ratio = 0.05,
                 stage1 = True,
                 k = None, # test number
                 *args, **kwargs):
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
            *args, **kwargs
        )
        self.mask_strategy = mask_strategy
        self.saved_path = saved_path
        os.makedirs(self.saved_path,exist_ok=True)
        self.m_ratio = m_ratio
        self.slide_test = slide_test
        self.max_rectangles = max_rectangles
        self.test_num=test_num
        self.val_mri_metric = {
            'AUPRC':[],
            'AUROC':[],
            'bestThresh':[],
            'bestDice':[],
            'diceScore':[]
        }
        self.test_mri_metric= {
            'AUPRC':[],
            'AUROC':[],
            'bestThresh':[],
            'bestDice':[],
            'diceScore':[]
        }
        self.mask_stage = mask_stage
        self.use_prior = use_prior
        self.metric_func = test_metrics
        self.metric_mean_func = arverage_metric
        self.val_bestThresh = 0
        self.per_metric_step = 50
        self.frq_mask_ratio = frq_mask_ratio
        self.rlfs_ratio = rlfs_ratio
        self.stage1 = stage1
        self.k = k

    def logkp(self, x, k=10000):
        return x
        # return torch.log(x + k)

    def rlogkp(self, x, k=10000):
        return x
        # return torch.exp(x) - k

    def get_learned_conditioning(self, c, ID=None, slices=None):
        mask_ratio = self.frq_mask_ratio
        N = len(c)
        
        cwt = self.first_stage_model.encode(c)
        cwt2 = self.get_first_stage_encoding(cwt).detach()
        c3 = torch.cat([cwt2, cwt2], dim=1)

        c0_high = getattr(self.cond_stage_model, 'get_low_frequency_single')(c, mask_ratio)
        c1_high_real = getattr(self.cond_stage_model, 'get_mri_prior_info_frq')(c0_high.real, mask_ratio, 'real')
        c1_high_imag = getattr(self.cond_stage_model, 'get_mri_prior_info_frq')(c0_high.imag, mask_ratio, 'imag')

        c0_norm_w_real = torch.linalg.norm(c0_high.real.flatten(1), dim=-1)
        c1_norm_w_real = torch.linalg.norm(self.rlogkp(c1_high_real).flatten(1), dim=-1)
        c01_scale_real = (c0_norm_w_real / c1_norm_w_real).reshape(N, 1, 1, 1)

        c0_norm_w_imag = torch.linalg.norm(c0_high.imag.flatten(1), dim=-1)
        c1_norm_w_imag = torch.linalg.norm(self.rlogkp(c1_high_imag).flatten(1), dim=-1)
        c01_scale_imag = (c0_norm_w_imag / c1_norm_w_imag).reshape(N, 1, 1, 1)

        c1_high_pi = torch.complex(self.rlogkp(c1_high_real) * c01_scale_real, self.rlogkp(c1_high_imag) * c01_scale_imag)

        c1_pi = getattr(self.cond_stage_model, 'get_combine_img_low')(c, c1_high_pi, mask_ratio)
        rm_high_freq = getattr(self.cond_stage_model, 'remove_high_frequency_single')(c1_pi , mask_ratio)
        
        
        c1 = self.first_stage_model.encode(c1_pi)
        c1 = self.get_first_stage_encoding(c1).detach()
        
        if self.stage1:
        ##### stage one ######
            c01_frq_low_real_loss = torch.nn.functional.l1_loss(self.rlogkp(c1_high_real).reshape(N, -1), c0_high.real.reshape(N, -1), reduction='none').mean(-1)
            c01_frq_low_imag_loss = torch.nn.functional.l1_loss(self.rlogkp(c1_high_imag).reshape(N, -1), c0_high.imag.reshape(N, -1), reduction='none').mean(-1)
            c3 = [None, c01_frq_low_real_loss, c01_frq_low_imag_loss]
        ###### stage one end ######
        else:
            # stage two or test #
            cwt = getattr(self.cond_stage_model, 'remove_low_frequency_single')(c, self.rlfs_ratio)
            c2wt = self.first_stage_model.encode(cwt)
            c2wt = self.get_first_stage_encoding(c2wt).detach()
            c3 = torch.cat([c1, c2wt], dim=1)
            if ID is not None and slices is not None:
                os.makedirs(os.path.join(self.saved_path,'c1_pi'),exist_ok=True)
                os.makedirs(os.path.join(self.saved_path,'ci_pi_rm_high_freq'),exist_ok=True)
                os.makedirs(os.path.join(self.saved_path,'cwt'),exist_ok=True)
                
                for i_ in range(len(ID)):
                    IDi = ID[i_]
                    os.makedirs(os.path.join(self.saved_path,'c1_pi',IDi.split('/')[-1]),exist_ok=True)
                    os.makedirs(os.path.join(self.saved_path,'ci_pi_rm_high_freq',IDi.split('/')[-1]),exist_ok=True)
                    os.makedirs(os.path.join(self.saved_path,'cwt',IDi.split('/')[-1]),exist_ok=True)
                    torch.save(c1_pi[i_],os.path.join(self.saved_path,'c1_pi',IDi.split('/')[-1],f'{slices}.pt'))
                    torch.save(rm_high_freq[i_],os.path.join(self.saved_path,'ci_pi_rm_high_freq',IDi.split('/')[-1],f'{slices}.pt'))
                    torch.save(cwt[i_],os.path.join(self.saved_path,'cwt',IDi.split('/')[-1],f'{slices}.pt'))
        
        return c3
        

    @torch.no_grad()
    def pixel_mask(self, x, m_ratio=0.5):
        B, _, H, W  = x.shape
        mask = torch.rand(1, 1, H, W)
        mask[mask>=m_ratio] = 1
        mask[mask<m_ratio] = 0
        return mask

    @torch.no_grad()
    def window_mask(self, x_start, m_ratio=0.2, max_rectangles=6):
        B, _, H, W  = x_start.shape
        mask = torch.ones((1, 1, H, W), dtype=torch.float32)
        num_rectangles = random.randint(0,max_rectangles)
        for _ in range(num_rectangles):
            # Randomly generate rectangle position and size
            size_ratio = random.uniform(0.01, m_ratio)
            rect_h = int(H * size_ratio)
            rect_w = int(W * size_ratio)
            y = random.randint(0, H - rect_h - 1)
            x = random.randint(0, W - rect_w - 1)
            mask[0, 0, y:y+rect_h, x:x+rect_w] = 0.0
        return mask

    @torch.no_grad()
    def slide_masks(self, x, m_ratio=0.2):
        B, _, H, W = x.shape
        mask = torch.ones((1, 1, H, W), dtype=torch.float32)
        mask_list = []
        for i in range(int(1/m_ratio)):
            for j in range(int(1/m_ratio)):
                mask_ = mask.clone()
                mask_[:,:,round(i*m_ratio*H):round((i+1)*m_ratio*H),round(j*m_ratio*W):round((j+1)*m_ratio*W)]=0
                mask_list.append(mask_.to(self.device))
        return mask_list  

    def forward(self, xs, c, *args, **kwargs):
        x, x_, mask_x_ = xs
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(xs, c, t, *args, **kwargs)

    @torch.no_grad()
    def get_input(self, batch, k, slide_mask=False, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, ID=None, slices=None):
        
        x = super(LatentDiffusion, self).get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        ### x or z mask step
        mask = torch.ones_like(x)
        x_ = x.clone()
        z_ = z.clone()
        if self.mask_stage=='img':
            x_ = x.clone()
            if self.mask_strategy == 'pixel':
                mask = self.pixel_mask(x_, m_ratio=self.m_ratio)
            elif self.mask_strategy == 'window':
                mask = self.window_mask(x_, m_ratio=self.m_ratio, max_rectangles=self.max_rectangles)
            if self.mask_strategy!='None':
                expanded_mask = mask.expand(x_.shape).to(self.device)
                x_[expanded_mask == 0] = -1
            encoder_posterior = self.encode_first_stage(x_)
            z_ = self.get_first_stage_encoding(encoder_posterior).detach()
        elif self.mask_stage=='latent':
            z_ = z.clone()
            if self.mask_strategy == 'pixel':
                mask = self.pixel_mask(z_, m_ratio=self.m_ratio)
            elif self.mask_strategy == 'window':
                mask = self.window_mask(z_, m_ratio=self.m_ratio, max_rectangles=self.max_rectangles)
            if self.mask_strategy!='None':
                z_ = z_ * mask.to(self.device)
        elif self.mask_stage=='None':
            mask = torch.ones_like(x)
            x_ = x.clone()
            z_ = z.clone()
        else:
            raise NotImplementedError()
        ###
        # # TODO L10 ing
        # x_ = x.clone()
        # z_ = z.clone()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            
            if self.mask_stage=='img' and self.mask_strategy!='None':
                expanded_mask = mask.expand(xc.shape).to(self.device)
                xc[expanded_mask == 0] = -1

            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc,ID=ID,slices=slices)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device),ID=ID,slices=slices)
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}
        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}

        c_list = []
        if (self.mask_stage=='latent' and self.mask_strategy!='None') and (not self.cond_stage_trainable or force_c_encode):
            if not slide_mask:
                if self.model.conditioning_key != 'hybrid':
                    c = cond * mask.to(self.device)
                elif self.model.conditioning_key == 'hybrid' and self.use_prior:
                    c['c_concat'][0] = c['c_concat'][0] * mask.to(self.device)
                    c['c_crossattn'][0] = c['c_crossattn'][0] * mask.to(self.device)
                else:
                    raise NotImplementedError()
            else:
                mask_list = self.slide_masks(c, self.m_ratio)
                c_list = [c*m for m in mask_list]
        
        if not slide_mask:
            out = [(z, z_, mask), c]
        else:
            out = [(z, z_, mask), c_list, mask_list]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    def p_losses(self, x_start, cond, t, noise=None):

        if self.stage1:
            cond, c01_frq_low_real_loss, c01_frq_low_imag_loss = cond
            loss_dict = {}
            loss_dict.update({f'train/loss_c01_frq_real_low': c01_frq_low_real_loss.mean()})
            # loss_dict.update({f'train/loss_c01_frq_imag_low': c01_frq_low_imag_loss.mean()})
            loss = c01_frq_low_real_loss.mean()
            # loss = c01_frq_low_real_loss.mean() + c01_frq_low_imag_loss.mean()
            return loss, loss_dict

        # cond, c01_frq_low_loss, c01_frq_middle_loss = cond
        # loss_dict = {}
        # loss_dict.update({f'train/loss_c01_frq_low': c01_frq_low_loss.mean()})
        # loss_dict.update({f'train/loss_c01_frq_middle': c01_frq_middle_loss.mean()})
        # # loss = c01_frq_low_loss.mean()
        # # loss = c01_frq_middle_loss.mean()
        # loss = c01_frq_low_loss.mean() + c01_frq_middle_loss.mean()
        # return loss, loss_dict

        x_start, x_start_, mask_x_ = x_start
        noise = default(noise, lambda: torch.randn_like(x_start_))
        x_noisy = self.q_sample(x_start=x_start_, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict
 
            
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        data = batch[self.first_stage_key]
        ind = data.shape[-1]//2
        batch[self.first_stage_key] = data[..., ind:ind+1]
        super().validation_step(batch, batch_idx)
        # # L10 add
        # if (self.current_epoch + 1) % self.per_metric_step == 0:
        #     batch[self.first_stage_key] = data
        #     self.validation_mertic_step(batch, batch_idx)


    @torch.no_grad()
    def validation_mertic_step(self, batch, batch_idx):
        data = batch[self.first_stage_key]
        if 'seg' in batch.keys():
            GT_volume = batch['seg']
        else:
            GT_volume = None
        ind = random.randint(1,data.shape[-1])
        cur_x_start = data[..., ind:ind+1]
        batch[self.first_stage_key] = cur_x_start

        N = data.shape[0]
        zs, c= self.get_input(batch, self.first_stage_key,
                        return_first_stage_outputs=False,
                        force_c_encode=True,
                        return_original_cond=False)

        samples,_ = self.sample_log(cond=c, batch_size=data.shape[0], ddim=False,
                                        ddim_steps=200, eta=1, x0=None, mask=None, noise_dropout=0.0)

        x_samples = self.decode_first_stage(samples).permute(0, 2, 3, 1).cpu()
        if GT_volume is not None:
            self.get_val_res(x_samples, cur_x_start, data_seg=GT_volume[..., ind:ind+1])
        else:
            self.get_val_res(x_samples, cur_x_start)

    def get_val_res(self, result, target, data_seg=None):
        target = target.cpu()
        data_mask = torch.where(target > -1, 1, 0)
        if data_seg is None:
            data_seg = torch.zeros_like(target)
        else:
            data_seg = data_seg.cpu()
        # calculate the residual image
        if True: # l1 or l2 residual
            diff_volume = torch.abs(target - result)
        else:
            diff_volume = (target - result)**2
        diff_volume = diff_volume.cpu()
        data_mask = data_mask.cpu()
        data_seg = data_seg.cpu()
            
    def test_save_vae_embed(self, batch, batch_idx):
        data = batch[self.first_stage_key]
        x = super(LatentDiffusion, self).get_input(batch, self.first_stage_key)
        x = x.to(self.device)
        N, C, H, W = x.shape
        embed_list = []
        for ind in range(C):
            c0 = getattr(self.cond_stage_model, 'remove_high_frequency_single')(x[:, ind:ind+1], 0.05)
            encoder_posterior = self.encode_first_stage(c0)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            embed_list.append(z)
        embed = torch.stack(embed_list, dim=1)
        for ind in range(N):
            torch.save(embed[ind].cpu(), batch['file_path'][ind].replace('t2-hd-bet', 'vae-high-embed').replace('_t2', '_vae'))

    def test_save_high_embed(self, batch, batch_idx):
        data = batch[self.first_stage_key]
        x = super(LatentDiffusion, self).get_input(batch, self.first_stage_key)
        x = x.to(self.device)
        N, C, H, W = x.shape
        embed_list = []
        for ind in range(C):
            c0 = getattr(self.cond_stage_model, 'get_high_frequency_single')(x[:, ind:ind+1])
            embed_list.append(c0)
        embed = torch.cat(embed_list, dim=1)
        for ind in range(N):
            torch.save(embed[ind].cpu(), batch['file_path'][ind].replace('t2-hd-bet', 't2-hf-all').replace('_t2', '_hf'))

    def test_save_prior_info(self):
        ccc = []
        aaa = getattr(self.cond_stage_model, 'get_mri_prior_info_mode')(torch.Tensor(1)).reshape(-1, 4, 32, 32)
        for i in range(aaa.shape[0]//4):
            bbb = self.decode_first_stage(aaa[i*4:(i+1)*4]).permute(0, 2, 3, 1).cpu()
            ccc.append(bbb)
        ccc = torch.concat(ccc, dim=0)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # self.test_save_high_embed(batch, batch_idx)
        # return None
        # self.test_save_prior_info()
        data = batch[self.first_stage_key]
        ID = batch["file_path"]
        if 'seg' in batch.keys():
            GT_volume = batch['seg']
        else:
            GT_volume = None
        rec_list = []
        mask_list = []
        indsl = 0
        indsr = data.shape[-1]
        for ind in range(indsl, indsr):
            cur_x_start = data[..., ind:ind+1]
            batch[self.first_stage_key] = cur_x_start
            N = data.shape[0]
            if self.slide_test:
                zs, c_list, m_list = self.get_input(batch, self.first_stage_key,
                            slide_mask=True,
                            return_first_stage_outputs=False,
                            force_c_encode=True,
                            return_original_cond=False,ID=ID,slices=ind)
                z, _, mask_z_ = zs
                samples_all = torch.zeros_like(z)
                for ci in range(len(c_list)):
                    # samples, _ = self.sample_log(cond=c_list[ci], batch_size=data.shape[0], ddim=False,
                    #                             ddim_steps=200, eta=1, x0=None, mask=None, noise_dropout=0.0)
                    samples, _ = self.sample_log(cond=c_list[ci], batch_size=data.shape[0], ddim=False,
                                                ddim_steps=200, eta=1, x0=z[:data.shape[0]], mask=None, noise_dropout=0.0)
                    samples_all = samples_all + samples * m_list[ci]
                x_samples = self.decode_first_stage(samples_all).permute(0, 2, 3, 1)
            else:
                zs, c = self.get_input(batch, self.first_stage_key,
                        return_first_stage_outputs=False,
                        force_c_encode=True,
                        return_original_cond=False,ID=ID,slices=ind)
                z, _, mask_z_ = zs
                # z = getattr(self.cond_stage_model, 'remove_low_frequency')(z, 0.05, False)
                samples, _ = self.sample_log(cond=c, batch_size=data.shape[0], ddim=False,
                                                ddim_steps=200, eta=1, x0=z[:data.shape[0]], mask=None, noise_dropout=0.0)
                # x_samples = self.decode_first_stage(z).permute(0, 2, 3, 1).cpu()
                x_samples = getattr(self.cond_stage_model, 'enhance_frq_result_compare')(cur_x_start.permute(0, 3, 1, 2), \
                                    self.decode_first_stage(samples), 0.10 if True else self.frq_mask_ratio).permute(0, 2, 3, 1).cpu()
                # x_samples = self.decode_first_stage(samples).permute(0, 2, 3, 1).cpu()
                x_samples = torch.where(cur_x_start.cpu().float() > torch.tensor(-1.0).float().cpu(), x_samples, torch.tensor(-1.0).float().cpu()).clamp(torch.tensor(-1.0).float().cpu(), torch.tensor(1.0).float().cpu())
                mask = torch.where(cur_x_start.float() > torch.tensor(-1.0).float().to(cur_x_start.device), torch.tensor(1.0).float().to(cur_x_start.device), torch.tensor(0.0).float().to(cur_x_start.device)).cpu()
                mask_list.append(mask)
                # x_samples = x_samples * mask.cpu()
                rec_list.append(x_samples)
            # if random.randint(0,100)>97:
                self.get_test_res(x_samples, cur_x_start, ID, ind, vis_save_path=self.saved_path, cond_mask=mask_z_,data_seg=GT_volume)
        rec_volume = torch.cat(rec_list,dim=-1)
        mask_volume = torch.cat(mask_list,dim=-1)
        diff_volume = torch.abs(data[..., indsl:indsr].detach().cpu() - rec_volume)
        self.get_test_metric(ID, data.shape[-1], \
            vis_save_path=self.saved_path, diff_volume=diff_volume, mask_volume=mask_volume, data_seg=GT_volume[..., indsl:indsr])

    # @torch.no_grad()
    # def test_step(self, batch, batch_idx):
    #     data = batch[self.first_stage_key]
    #     ID = batch["file_path"]
    #     if 'seg' in batch.keys():
    #         GT_volume = batch['seg']
    #     else:
    #         GT_volume = None
    #     rec_list = []
    #     mask_list = []
    #     diff_list = []
    #     indsl = 0
    #     indsr = data.shape[-1]
    #     for ind in range(indsl, indsr):
    #         rec_slice_list = []
    #         mask_slice_list = []
    #         diff_slice_list = []
    #         for i in range(len(ID)):
    #             rec_slice_list.append(torch.load(os.path.join(self.saved_path, "rec", ID[i].split('/')[-1].replace(".nii",f"_{ind}_rec.pt"))))
    #             mask_slice_list.append(torch.load(os.path.join(self.saved_path, "mask", ID[i].split('/')[-1].replace(".nii",f"_{ind}_mask.pt"))))
    #             diff_slice_list.append(torch.load(os.path.join(self.saved_path, "diff", ID[i].split('/')[-1].replace(".nii",f"_{ind}_diff.pt"))))
    #         rec_list.append(torch.stack(rec_slice_list, dim=0))
    #         mask_list.append(torch.stack(mask_slice_list, dim=0))
    #         diff_list.append(torch.stack(diff_slice_list, dim=0))
    #     rec_volume = torch.cat(rec_list, dim=-1)
    #     mask_volume = torch.cat(mask_list, dim=-1)
    #     diff_volume = torch.cat(diff_list, dim=-1)
    #     self.get_test_metric(ID, data.shape[-1], \
    #         vis_save_path=self.saved_path, diff_volume=diff_volume, mask_volume=mask_volume, data_seg=GT_volume[..., indsl:indsr])

    @torch.no_grad()
    def get_test_res(self, result, target, ID, ind, vis_save_path, cond_mask=None, data_seg=None, prior_sample=None):
        data_mask = torch.where(target > -1, 1, 0)
        target = target.cpu()
        result = result.cpu()
        # calculate the residual image
        if True: # l1 or l2 residual
            diff_volume = torch.abs(target - result)
        else:
            diff_volume = (target - result)**2
        diff_volume = diff_volume.cpu()
        data_mask = data_mask.cpu()
        if data_seg is not None:
            data_seg = data_seg.cpu()
        self.save_slices(data_mask,diff_volume,target,result,ID,ind,\
            vis_save_path=vis_save_path, cond_mask=cond_mask, data_seg=data_seg, prior_sample=prior_sample)
    
    def save_slices(self,data_mask,diff_volume, target, result, ID, ind, vis_save_path, cond_mask=None, data_seg=None, prior_sample=None):
        for i in range(len(result)):
            os.makedirs(os.path.join(vis_save_path,"mask"), exist_ok=True)
            
            print(ID[i].split('/')[-1].replace(".nii",f"_{ind}_mask.pt"))
            torch.save(data_mask[i],os.path.join(vis_save_path,"mask",ID[i].split('/')[-1].replace(".nii",f"_{ind}_mask.pt")))
            if data_seg is not None:
                os.makedirs(os.path.join(vis_save_path,"seg"), exist_ok=True)
                torch.save(data_seg[i,:,:,ind].detach().cpu().bool(),os.path.join(vis_save_path,"seg",ID[i].split('/')[-1].replace(".nii",f"_{ind}_seg.pt")))
            if cond_mask is not None:
                os.makedirs(os.path.join(vis_save_path,"cond"), exist_ok=True)
                torch.save(cond_mask[0][0].detach().cpu(),os.path.join(vis_save_path,"cond",ID[i].split('/')[-1].replace(".nii",f"_{ind}_cond.pt")))
            if prior_sample is not None:
                os.makedirs(os.path.join(vis_save_path,"prior"), exist_ok=True)
                torch.save(prior_sample, os.path.join(vis_save_path,"prior",ID[i].split('/')[-1].replace(".nii",f"_{ind}_prior.pt")))
            os.makedirs(os.path.join(vis_save_path,"diff"), exist_ok=True)
            torch.save(diff_volume[i], os.path.join(vis_save_path,"diff",ID[i].split('/')[-1].replace(".nii",f"_{ind}_diff.pt")))
            os.makedirs(os.path.join(vis_save_path,"target"), exist_ok=True)
            torch.save(target[i], os.path.join(vis_save_path,"target",ID[i].split('/')[-1].replace(".nii",f"_{ind}_target.pt")))
            os.makedirs(os.path.join(vis_save_path,"rec"), exist_ok=True)
            torch.save(result[i], os.path.join(vis_save_path,"rec",ID[i].split('/')[-1].replace(".nii",f"_{ind}_rec.pt")))
        print("SAVED!")
   
    def get_test_metric(self, ID, slice_len, vis_save_path, diff_volume, mask_volume, data_seg):
        # raw_data_seg = data_seg
        print("LOAD to Test metric!")
        for i in range(len(ID)):
            self.test_mri_metric = self.metric_func(ID[i], vis_save_path, self.test_mri_metric,data_seg[i].cpu(),diff_volume[i],mask_volume[i],val_bestThresh=self.val_bestThresh)
        with open(os.path.join(self.saved_path,'test_metrics.log'), 'a') as f:
            f.write(f'{ID}\n')
            f.write(f'{self.metric_mean_func(self.test_mri_metric)}\n')
            f.write("****************************************\n")
