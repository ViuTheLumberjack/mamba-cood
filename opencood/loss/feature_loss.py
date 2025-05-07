# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim  # Install via: pip install pytorch-msssim


class FeatureLoss(nn.Module):
    def __init__(self, args):
        super(FeatureLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.loss_dict = {}

    def aggressive_ssim_loss(self, output, target, alpha=1.0):
        ssim_val = ssim(output, target, data_range=1.0, size_average=True)
        diff_map = torch.abs(output - target)
        weight_map = torch.exp(alpha * diff_map)
        weighted_l1_loss = (weight_map * diff_map).mean()
        return (1 - ssim_val) + weighted_l1_loss  
    
    def local_mask(self, input_data, output, target):

        threshold = 1.0 #todo find th
        diff_mask = (target - input_data).abs() > threshold  # e.g., 0.01
        loss = F.l1_loss(output[diff_mask], target[diff_mask])

        return loss

    def l1_loss(self, output, target):
        return F.l1_loss(output, target)
    
    def total_variation(self, x):
        """
        x: Tensor of shape [B, C, H, W]
        Returns scalar TV loss
        """
        tv_h = torch.mean((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2)
        tv_w = torch.mean((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2)
        return tv_h + tv_w

    def l1_ssim(self, input_data, output, target):
        l1 = F.l1_loss(output, target)

        # SSIM loss
        ssim_loss = 1 - ssim(output, target, data_range=1.0)

        # Total variation regularizer on delta
        delta = output - input_data
        tv = self.total_variation(delta)

        # Combined loss
        loss = l1 + 0.5 * ssim_loss + 0.1 * tv
        return loss
    

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """

        feature_base = output_dict['feature_base']
        feature_pred = output_dict['feature_output']
        feature_gt = target_dict['ego']['gt_features']
        record_len = target_dict['ego']['record_len']

        loss = 0
        for i in range(len(record_len)):
            base = feature_base[i, :record_len[i]]
            pred = feature_pred[i, :record_len[i]]
            gt = feature_gt[i, :record_len[i]]

            #smooth L1 loss
            # loss += F.smooth_l1_loss(pred, gt)

            # L1 loss
            loss += self.l1_loss(pred, gt)


            # loss += self.local_mask(base, pred, gt)
            # loss += self.l1_loss(pred, gt)
            # loss += self.l1_ssim(base, pred, gt)
            # loss += 1 - ssim(pred, gt, data_range=gt.max(), size_average=True)
            # loss += self.aggressive_ssim_loss(pred, gt, alpha=1.0)
        loss = loss / len(record_len)

        self.loss_dict.update({'total_loss': loss})

        return loss



    def logging(self, epoch, batch_id, batch_len, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || " % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f ||" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item()))
