# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import einops


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss
    
# -------------- SSIM and MS-SSIM implementation from https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py --------------
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

def _init_delay_loss(delay_type, delay_arg):
        match delay_type:
            case 'huber':
                return nn.HuberLoss(reduction='none', beta=delay_arg)
            case 'smooth_l1':
                return nn.SmoothL1Loss(reduction='none', beta=delay_arg)
            case 'charbonnier':
                return CharbonnierLoss(reduction='none', eps=delay_arg)
            case "mse":
                return nn.MSELoss(reduction='none')
            case "l1":
                return nn.L1Loss(reduction='none')

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1) implementation.
    Args:
        eps: A small value to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = 'none') -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        loss = torch.sqrt(diff * diff + self.eps)

        match self.reduction:
            case 'mean':
                return loss.mean()
            case 'sum':
                return loss.sum()
            case 'none':
                return loss
            case _:
                raise ValueError(f"Invalid reduction mode: {self.reduction}. Supported modes are 'none', 'mean', and 'sum'.")

class FeatureMapPredictionLoss(nn.Module):
    def __init__(self, args):
        super(FeatureMapPredictionLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0
        self.args = args
        self.freeze_heads = args['freeze_heads']

        self.cls_weight = args['cls_weight'] # lamda cls
        self.reg_coe = args['reg'] # lamda reg

        self.delay_type = args.get('delay_type', 'charbonnier') # delay loss type
        self.delay_arg = args.get('delay_arg', 0.1) # delay loss beta
        self.delay_weight = args.get('delay_weight', 1.0) # lamda delay
        self.delay_loss_func = _init_delay_loss(self.delay_type, self.delay_arg)
        self.normalized = args.get('normalized', True) # whether to normalize the loss

        self.fg_weight = args.get('fg_weight', 10.0) # lamda foreground
        self.bg_weight = args.get('bg_weight', 1.0) # lamda
        self.temp_weight = args.get('temp_weight', 5.0) # lamda temporal
        self.cosine_weight = args.get('cosine_weight', 0.5) # lamda cosine similarity

        self.loss_dict = {}            

    def delay_loss(self, output, target, weight=None):
        """
        Parameters
        ----------
        output : torch.Tensor size (T*B, C, H, W)
        target : torch.Tensor size (T*B, C, H, W)

        Returns
        -------
        loss : float
        """
        delay_loss = self.delay_loss_func(output, target)

        if weight is not None:
            delay_loss = delay_loss * weight
        
        return delay_loss
        
    def forward(self, feature_pred, predictions, target_dict, train=True):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        
        target_dict_copy = target_dict.copy()
        target_dict = target_dict['ego']['label_dict']
        #added loss
        past = target_dict_copy['ego']['past_features']
        current = target_dict_copy['ego']['current_features']
        feature_gt = target_dict_copy['ego']['gt_features']
        record_len = target_dict_copy['ego']['record_len']

        ego_list = target_dict_copy['ego']['ego_list']
        
        loss = 0
        
        #print(ego_flag)

        B, T, C, H, W = predictions.shape
        pred = predictions #[:, i]
        gt = feature_gt #[:, i]
        current = current.unsqueeze(1).repeat(1, T, 1, 1, 1) # [B, T, C, H, W]

        if not train:
            for i in range(len(record_len)):
                ego_flag = torch.Tensor(ego_list[i])
                # from 0 to 1, and from 1 to 0
                ego_flag = torch.where(ego_flag == 0, 1, 0)
            
            ego_flag = ego_flag.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, T, C, H, W).cuda()
            # print(f"ego_flag shape: {ego_flag.shape}")
            pred = pred * ego_flag
            gt = gt * ego_flag
            current = current * ego_flag

        #prepend currnet
        pred_2 = torch.cat([current, pred], dim=1)  # [B, T+1, C, H, W]
        gt_2 = torch.cat([current, gt], dim=1)

        pred_diff = pred_2[:, 1:] - pred_2[:, :-1]      # [B, T, C, H, W]
        gt_diff   = gt_2[:, 1:] - gt_2[:, :-1]         # [B, T, C, H, W]

        #print(pred.shape, gt.shape)
        pred = einops.rearrange(pred, 'b t ... -> (t b) ...')
        gt = einops.rearrange(gt, 'b t ... -> (t b) ...')
            
        delay_loss = self.delay_loss(pred, gt, weight=None)
        fg = (gt.abs().sum(dim=1, keepdim=True) > 1e-3).float()

        if not self.normalized:
            fg_loss = (delay_loss * fg).sum() / (fg.sum() + 1e-6)
            fg_loss = fg_loss * self.fg_weight
            bg_loss = (delay_loss * (1 - fg)).sum() / ((1 - fg).sum() + 1e-6)
            bg_loss = bg_loss * self.bg_weight
        else:
            # all-in-one fg and bg calculation
            # the weight is calculated based on the gt, so that the loss is normalized by the gt magnitude
            gt_norm = gt / (gt.detach().amax(dim=(-2, -1), keepdim=True) + 1e-6)
            w = 0.1 + 2.0 * (gt.abs().sum(1, keepdim=True)).float() + 20.0 * gt_norm.pow(2)
            
            delay_loss = (delay_loss * w).sum() / (w.sum() + 1e-6)
            fg_loss = delay_loss
            bg_loss = torch.tensor(0.0).cuda()

        # Foreground mask: active in any timestep
        if self.temp_weight > 0:
            fg_temp = (gt_diff.abs().sum(2, keepdim=True) > 1e-3).float()
            temp_loss = self.delay_loss(pred_diff, gt_diff, weight=None)
            temp_fg_loss = (temp_loss * fg_temp).sum() / (fg_temp.sum() + 1e-6)
            temp_bg_loss = (temp_loss * (1 - fg_temp)).sum() / ((1 - fg_temp).sum() + 1e-6)

            temp_loss = self.fg_weight * temp_fg_loss + self.bg_weight * temp_bg_loss
            temp_loss = temp_loss * self.temp_weight
        else:
            temp_loss = torch.tensor(0.0).cuda()

        if self.cosine_weight > 0:
            l_cos = F.cosine_similarity(pred * fg, gt * fg, dim=1)
            l_cos = 1 - (l_cos.sum() / (fg.sum() + 1e-6))
            l_cos = l_cos * self.cosine_weight
        else:
            l_cos = torch.tensor(0.0).cuda()
        
        loss = fg_loss + bg_loss + temp_loss + l_cos
        loss = loss * self.delay_weight
        
        self.loss_dict.update({
                               'loss_feature': loss, 
                               'foreground_loss': fg_loss,
                               'foreground_ratio': fg.sum() / (fg.sum() + (1 - fg).sum()),
                               'background_loss': bg_loss,
                               'temporal_loss': temp_loss,
                               'residual_loss': l_cos,
                               'predictions_mean': predictions.mean(),
                               'predictions_std': predictions.std()
                               })

        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None):
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
        pred_loss = self.loss_dict['loss_feature']
        mu = self.loss_dict['predictions_mean']
        std = self.loss_dict['predictions_std']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || %.4f - %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    pred_loss.item(), mu.item(), std.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || %.4f - %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    pred_loss.item(), mu.item(), std.item()))
                