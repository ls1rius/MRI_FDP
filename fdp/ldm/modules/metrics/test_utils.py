import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from skimage.measure import regionprops, label
from torchvision.transforms import ToTensor, ToPILImage
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
# import monai
from torch.nn import functional as F
# from PIL import Image
from tqdm import tqdm 
import matplotlib.colors as colors

def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    
    if True:
        for prop in props:
            if prop['filled_area'] <= 7:
                volume[cc_volume == prop['label']] = 0
    else:
        max_volume = 0
        max_label = -1
        for prop in props:
            if prop['area']>10 and prop['area'] > max_volume:
                max_volume = prop['area']
                max_label = prop['label']
        volume[cc_volume != max_label] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume

def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  #x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    #print(str(np.mean(x>q_bottom)) + str(np.mean(y)))
    val_top = dice(x > q_top, y)
    #print(str(np.mean(x>q_top)) + str(np.mean(y)))
    #val_bottom = val_fn(x, y, q_bottom) # val_fn is the dice calculation dice(p, g)
    #val_top = val_fn(x, y, q_top)

    if val_bottom >= val_top:
        if val_bottom > max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top > max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val,max_point=max_point)

def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum + 1e-12)
    return score

    
def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions, pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds   

# Dice Score 
def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tp)


def normalize(tensor): # THanks DZimmerer
    tens_deta = tensor.detach().cpu()
    tens_deta -= float(np.min(tens_deta.numpy()))
    tens_deta /= float(np.max(tens_deta.numpy()))

    return tens_deta

def apply_brainmask(x, brainmask, erode , iterations):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=iterations)
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))

def apply_brainmask_volume(vol,mask_vol, erode=True, iterations=10) : 
    vol = vol.squeeze()
    for s in range(vol.shape[-1]): 
        slice = vol[:,:,s]
        mask_slice = mask_vol[:,:,s]
        eroded_vol_slice = apply_brainmask(slice, mask_slice, erode=erode, iterations=vol.shape[1]//25)
        vol[:,:,s] = eroded_vol_slice
    return vol

def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume


@torch.no_grad()
def test_metrics(filename, vis_save_path, mri_mertic,data_seg,diff_volume,data_mask,val_bestThresh=None):
    diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.cpu())
    diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy(), kernelsize=5)) # bring back to tensor

    AUPRC, _precisions, _recalls, _threshs = compute_prc(diff_volume.squeeze().flatten(), np.array(data_seg.squeeze().flatten()).astype(bool))
    AUROC, _fpr, _tpr, _threshs = compute_roc(diff_volume.squeeze().flatten(), np.array(data_seg.squeeze().flatten()).astype(bool))
    ######
    # gready search for threshold
    bestDice, bestThresh = find_best_val(diff_volume.numpy().flatten(),  # threshold search with a subset of EvaluationSet
                                        data_seg.numpy().flatten().astype(bool),
                                        val_range=(0, diff_volume.numpy().max()),
                                        max_steps=10,
                                        step=0,
                                        max_val=0,
                                        max_point=diff_volume.numpy().max())
    if val_bestThresh:
        diffs_thresholded = diff_volume > val_bestThresh
    else:
        diffs_thresholded = diff_volume > bestThresh
    # ### add for test draw img
    # auprc_num_list = []
    # for ind in range(diff_volume.shape[-1]):
    #     auprc_num, _, _, _ = compute_prc(diffs_thresholded[..., ind].flatten().cpu().numpy(), data_seg[..., ind].flatten().numpy().flatten().astype(bool))
    #     auprc_num_list.append(auprc_num)
    # torch.save(torch.Tensor(auprc_num_list), os.path.join(vis_save_path, os.path.basename(filename).split('.')[0] + '_auprc_slice.pth'))
    # ###
    diffs_thresholded = filter_3d_connected_components(np.squeeze(diffs_thresholded))
    diceScore = dice(diffs_thresholded.cpu().numpy(), data_seg.numpy().flatten().astype(bool))
    # add for save by L10
    os.makedirs(os.path.join(vis_save_path, 'res'), exist_ok=True)
    torch.save(diffs_thresholded, os.path.join(vis_save_path, 'res', 
                os.path.basename(filename).replace('t2', 'res').replace('.nii.gz', '.pt').replace('.nii', '.pt')))
    ###
    mri_mertic["AUPRC"].append(AUPRC)
    mri_mertic["AUROC"].append(AUROC)
    if not val_bestThresh:
        mri_mertic["bestThresh"].append(bestThresh)
    mri_mertic["bestDice"].append(bestDice)
    mri_mertic["diceScore"].append(diceScore)
    return mri_mertic


# @torch.no_grad()
# def test_metrics(filename, vis_save_path, mri_mertic,data_seg,diff_volume,data_mask,val_bestThresh=None):
#     if val_bestThresh:
#         diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.cpu())
#         diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy(), kernelsize=5)) # bring back to tensor
#     ######
#     # diff_volume = diff_volume * data_mask
#     if True:
#         # gready search for threshold
#         bestDice, bestThresh = find_best_val(diff_volume.numpy().flatten(),  # threshold search with a subset of EvaluationSet
#                                             data_seg.numpy().flatten().astype(bool),
#                                             val_range=(0, diff_volume.numpy().max()),
#                                             max_steps=10,
#                                             step=0,
#                                             max_val=0,
#                                             max_point=diff_volume.numpy().max())
#         if val_bestThresh:
#             diffs_thresholded = diff_volume > val_bestThresh
#         else:
#             diffs_thresholded = diff_volume > bestThresh
#     else:
#         diffs_thresholded_list_ = []
#         bestDice_sum = 0.0
#         bestThresh_sum = 0.0
#         for ind in range(diff_volume.shape[-1]):
#             # if ind == 100:
#             #     # torch.save(data_seg[..., ind:ind+1].cpu(), './ttmp.pth')
#             #     print('hello')
#             bestDice_, bestThresh_ = find_best_val(diff_volume[..., ind:ind+1].numpy().flatten(),  # threshold search with a subset of EvaluationSet
#                                         data_seg[..., ind:ind+1].numpy().flatten().astype(bool),
#                                         val_range=(0, diff_volume[..., ind:ind+1].numpy().max()),
#                                         max_steps=10,
#                                         step=0,
#                                         max_val=0,
#                                         max_point=diff_volume[..., ind:ind+1].numpy().max())
#             diffs_thresholded_ = diff_volume[..., ind:ind+1] > bestThresh_
#             diffs_thresholded_list_.append(diffs_thresholded_)
#             bestDice_sum += bestDice_
#             bestThresh_sum += bestThresh_
#         diffs_thresholded = torch.cat(diffs_thresholded_list_, dim=2)
#         bestDice = bestDice_sum / diff_volume.shape[-1]
#         bestThresh = bestThresh_sum / diff_volume.shape[-1]
#     ######
#     structure = np.ones((3, 3))
#     diffs_thresholded = diffs_thresholded.cpu().numpy()
#     for s in range(diffs_thresholded.shape[-1]): 
#         diffs_thresholded[:,:,s] = scipy.ndimage.binary_opening(diffs_thresholded[:,:,s], structure=structure)
#     diffs_thresholded = filter_3d_connected_components(np.squeeze(diffs_thresholded))
#     diceScore = dice(diffs_thresholded, data_seg.numpy().flatten().astype(bool))
#     # # add for save L10
#     # os.makedirs(os.path.join(vis_save_path, 'res'), exist_ok=True)
#     # torch.save(diffs_thresholded, os.path.join(vis_save_path, 'res', 
#     #             os.path.basename(filename).replace('t2', 'res').replace('.nii.gz', '.pt').replace('.nii', '.pt')))
#     # ###
#     AUROC, _fpr, _tpr, _threshs = compute_roc(diff_volume.squeeze().flatten(), np.array(data_seg.squeeze().flatten()).astype(bool))
#     AUPRC, _precisions, _recalls, _threshs = compute_prc(diff_volume.squeeze().flatten(), np.array(data_seg.squeeze().flatten()).astype(bool))
#     # AnomalyScoreReco = [] # Reconstruction based Anomaly score
#     # for slice in range(diff_volume.shape[-1]):
#     #     score = diff_volume[...,slice][data_mask[...,slice]>0].mean()
#     #     # score = diff_volume[...,slice][(data_mask[...,slice]>0)&(diff_volume[...,slice]>bestThresh)].mean()
#     #     if score.isnan() : # if no brain exists in that slice
#     #         AnomalyScoreReco.append(0.0) 
#     #     else: 
#     #         AnomalyScoreReco.append(score)
#     # label = [] # store labels here
#     # for slice in range(data_seg.shape[-1]) :  #iterate through volume
#     #     if np.array(data_seg[...,slice]).astype(bool).any(): # if there is an anomaly segmentation
#     #         label.append(1) # label = 1
#     #     else :
#     #         label.append(0) # label = 0 if there is no Anomaly in the slice
#     # AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(AnomalyScoreReco), np.array(label))
#     # AUROC, _fpr, _tpr, _ = compute_roc(np.array(AnomalyScoreReco), np.array(label))

#     mri_mertic["AUPRC"].append(AUPRC)
#     mri_mertic["AUROC"].append(AUROC)
#     if not val_bestThresh:
#         mri_mertic["bestThresh"].append(bestThresh)
#     mri_mertic["bestDice"].append(bestDice)
#     mri_mertic["diceScore"].append(diceScore)
#     return mri_mertic

def arverage_metric(metric_dict, val_bestThresh):
    metric_dict["AUPRCMean"] = np.nanmean(metric_dict["AUPRC"])
    metric_dict["AUROCMean"] = np.nanmean(metric_dict["AUROC"])
    if not val_bestThresh:
        metric_dict["bestThreshMean"] = np.nanmean(metric_dict["bestThresh"])
    metric_dict["bestDiceMean"] = np.nanmean(metric_dict["bestDice"])
    metric_dict["diceScoreMean"] = np.nanmean(metric_dict["diceScore"])
    return metric_dict