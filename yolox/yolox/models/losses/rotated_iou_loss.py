#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from loguru import logger
import torch
import torch.nn as nn

try:
    from mmcv.ops import diff_iou_rotated_2d
except:  # noqa: E722
    diff_iou_rotated_2d = None

from .utils import weighted_loss
#
#
# class IOUloss(nn.Module):
#     def __init__(self, reduction="none", loss_type="iou"):
#         super(IOUloss, self).__init__()
#         self.reduction = reduction
#         self.loss_type = loss_type
#
#     def forward(self, pred, target):
#         assert pred.shape[0] == target.shape[0]
#
#         pred = pred.view(-1, 4)
#         target = target.view(-1, 4)
#         tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
#         br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
#
#         area_p = torch.prod(pred[:, 2:], 1)
#         area_g = torch.prod(target[:, 2:], 1)
#
#         en = (tl < br).type(tl.type()).prod(dim=1)
#         area_i = torch.prod(br - tl, 1) * en
#         area_u = area_p + area_g - area_i
#         iou = area_i / (area_u + 1e-16)
#
#         if self.loss_type == "iou":
#             loss = 1 - iou ** 2
#         elif self.loss_type == "giou":
#             c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
#             c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
#             area_c = torch.prod(c_br - c_tl, 1)
#             giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
#             loss = 1 - giou.clamp(min=-1.0, max=1.0)
#
#         if self.reduction == "mean":
#             loss = loss.mean()
#         elif self.reduction == "sum":
#             loss = loss.sum()
#
#         return loss


@weighted_loss
def rotated_iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        logger.warn('DeprecationWarning: Setting "linear=True" in poly_iou_loss is deprecated, please use "mode=`linear`" instead.')

    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')

    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious ** 2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


class RotatedIoULoss(nn.Module):
    """RotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, linear=False, eps=1e-6, reduction='mean', loss_weight=1.0, mode='log'):
        super(RotatedIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            logger.warn('DeprecationWarning: Setting "linear=True" in  IOULoss is deprecated, please use "mode=`linear`"  instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_iou_loss(pred, target, weight, mode=self.mode, eps=self.eps)
        return loss
