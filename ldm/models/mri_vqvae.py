from .autoencoder import VQModel
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
from torch.nn import functional as F
from ldm.modules.metrics.test_utils import test_metrics, arverage_metric


class MRIVQVAE(VQModel):
    def __init__(self,
                embed_dim,
                seg_key="seg",
                **kwargs
                ):
        super().__init__(
            embed_dim=embed_dim,
            **kwargs
        )
        self.gt_key = seg_key
        self.metric = test_metrics
        self.arverage_metric = arverage_metric
        self.eval_dict = {
            'DiceScorePerVol':[],
            'BestThresholdPerVol':[],
            'AUPRCPerVol':[]
        }

    def validation_step(self, batch, batch_idx):
        data = batch[self.image_key]
        ind = data.shape[-1]//2
        batch[self.image_key] = data[..., ind:ind+1]
        super().validation_step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        inputs_volume = self.get_input(batch, self.image_key)
        GT_volume = self.get_input(batch, self.gt_key)
        
        recons = []
        for i in range(inputs_volume.shape[1]):
            slice_i = inputs_volume[:,i:i+1,:,:]
            # print(slice_i.shape)
            reconstruction_i, posterior_i = self(slice_i)
            recons.append(reconstruction_i)
        final_volume = torch.cat(recons,dim=1)
        diff_volume = torch.abs(inputs_volume-final_volume)
       # Calculate Reconstruction errors with respect to anomal/normal regions
    
        self.eval_dict = self.metrics(self.eval_dict,final_volume,inputs_volume,GT_volume)
        

    @torch.no_grad()
    def on_test_epoch_end(self) :
    # average over all test samples
        self.eval_dict = self.arverage_metric(self.eval_dict)
        print(self.eval_dict)

class MRIVQModelInterface(MRIVQVAE):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec