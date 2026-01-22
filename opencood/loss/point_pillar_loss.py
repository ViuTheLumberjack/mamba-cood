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

class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0
        self.args = args
        self.freeze_heads = args['freeze_heads']

        self.cls_weight = args['cls_weight'] # lamda cls
        self.reg_coe = args['reg'] # lamda reg
        self.delay_beta = args['delay']
        self.delay_coeff = args.get('delay_coeff', 1.0) # lamda delay
        self.loss_dict = {}

    def delay_loss(self, output, target):
        """
        Parameters
        ----------
        output : torch.Tensor size (T, B, C, H, W)
        target : torch.Tensor size (T, B, C, H, W)

        Returns
        -------
        loss : float
        """
       # Assicuriamoci che input e target abbiano la stessa shape
        assert output.shape == target.shape
        
        # 1. Calcola l'intensità della Ground Truth per ogni pixel spaziale (B, H, W)
        # Usiamo einops per sommare il valore assoluto lungo i canali C
        target_magnitude = einops.reduce(torch.abs(target), '... c h w -> ... 1 h w', 'sum')

        # 2. Crea la maschera attiva (Foreground)
        # Se la somma delle feature in un pixel è > 1e-4, consideralo un oggetto/info utile
        active_mask = (target_magnitude > 1e-2).float()

        # 3. Definisci i pesi
        # Foreground: peso 1.0
        # Background: peso 0.1 (o 0.05 se vuoi penalizzarlo ancora meno)
        weights = active_mask * 2.0 + (1 - active_mask) * 0.25

        # 4. Calcola la differenza assoluta (L1)
        diff = torch.abs(output - target)

        # 5. Applica i pesi alla differenza
        weighted_diff = diff * weights

        # 6. Calcola la media (o somma normalizzata per numero di elementi)
        # Riduciamo tutto a un singolo scalare
        # loss = einops.reduce(weighted_diff, '... -> 1', 'mean')

        # Rimuovi la dimensione extra dello scalare
        #print("Delay loss:", weighted_diff.shape)
        return weighted_diff
    
        #return F.l1_loss(output, target, reduction='none')
        return F.huber_loss(output, target, reduction='none', delta=self.delay_beta)
        # Charbonnier loss
        def charbonnier_loss(x, epsilon=1e-6):
            return torch.sqrt(x * x + epsilon)
        
        diff = output - target
        loss = charbonnier_loss(diff, self.delay_beta)

        def cosine_similarity_loss(x: torch.Tensor, y: torch.Tensor):
            # T, C, H, W = x.shape
            x_flat = einops.rearrange(x, 't c h w -> t (c h w)')
            y_flat = einops.rearrange(y, 't c h w -> t (c h w)')

            cos = F.cosine_similarity(x_flat, y_flat, dim=1, eps=1e-6)
            cos = 1 - cos
            return cos 
        
        return loss #cosine_similarity_loss(output, target)

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm']
        psm = output_dict['psm']
        target_dict_copy = target_dict.copy()
        target_dict = target_dict['ego']['label_dict']
        targets = target_dict['targets']

        #added loss
        feature_pred = output_dict['feature_output']
        predictions = output_dict['predictions']
        feature_gt = target_dict_copy['ego']['gt_features']
        record_len = target_dict_copy['ego']['record_len']
        ego_list = target_dict_copy['ego']['ego_list']

        if self.args['module_delay']:
            loss = 0
            for i in range(len(record_len)):
                ego_flag = torch.Tensor(ego_list[i])
                #from 0 to 1, and from 1 to 0
                ego_flag = torch.where(ego_flag == 0, 1, 0)

                pred = predictions[:, i, :record_len[i]]
                gt = feature_gt[:, i, :record_len[i]]

                delay_loss = self.delay_loss(pred, gt)

                ego_flag = ego_flag.unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(delay_loss.shape[0], 1, delay_loss.shape[2], delay_loss.shape[3]).cuda()
                # print(ego_flag.shape, delay_loss.shape)

                try:
                    delay_loss = delay_loss * ego_flag
                except:
                    print('error in l1 loss')
                delay_loss = delay_loss.sum() / (ego_flag.sum() + 1e-6)

                loss += delay_loss

            loss = loss / len(record_len)
            loss *= self.delay_coeff
        else:
            loss = torch.zeros(1).cuda()

        if self.args['freeze_heads'] == False:
            conf_loss = 0
            cls_preds = psm.permute(0, 2, 3, 1).contiguous()

            box_cls_labels = target_dict['pos_equal_one']
            box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

            positives = box_cls_labels > 0
            negatives = box_cls_labels == 0
            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels
            cls_targets = cls_targets.unsqueeze(dim=-1)

            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), 2,
                dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            cls_preds = cls_preds.view(psm.shape[0], -1, 1)
            one_hot_targets = one_hot_targets[..., 1:]

            cls_loss_src = self.cls_loss_func(cls_preds,
                                            one_hot_targets,
                                            weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / psm.shape[0]
            conf_loss += cls_loss * self.cls_weight

            reg_loss = 0
        
            # regression
            rm = rm.permute(0, 2, 3, 1).contiguous()
            rm = rm.view(rm.size(0), -1, 7)
            targets = targets.view(targets.size(0), -1, 7)
            box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                    targets)
            loc_loss_src =\
                self.reg_loss_func(box_preds_sin,
                                reg_targets_sin,
                                weights=reg_weights)
            reg_loss_ = loc_loss_src.sum() / rm.shape[0]
            reg_loss_ *= self.reg_coe
            reg_loss += reg_loss_

            total_loss = reg_loss + conf_loss

            #if False:
            total_loss += loss
        else:
            total_loss = loss
            reg_loss = torch.zeros(1).cuda()
            conf_loss = torch.zeros(1).cuda()

        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss,
                               'loss_feature': loss, 
                               'predictions_mean': predictions.mean(),
                               'predictions_std': predictions.std()
                               })

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


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
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        mu = self.loss_dict['predictions_mean']
        std = self.loss_dict['predictions_std']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f || %.4f - %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), conf_loss.item(), reg_loss.item(),
                    mu.item(), std.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                  " || Loc Loss: %.4f || %.4f - %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), conf_loss.item(), reg_loss.item(),
                      mu.item(), std.item()))


        # writer.add_scalar('Regression_loss', reg_loss.item(),
        #                   epoch*batch_len + batch_id)
        # writer.add_scalar('Confidence_loss', conf_loss.item(),
        #                   epoch*batch_len + batch_id)