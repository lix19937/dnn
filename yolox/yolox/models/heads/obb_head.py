#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

# from yolox.utils import bboxes_iou
from ..losses import compute_kld_loss, KLDloss
# from yolox.utils import OBBOverlaps
# from .obb import PolyIOULoss
from yolox.models.components.network_blocks import BaseConv, DWConv
from ..losses import L1Loss, CELoss
from yolox.models.components.extra_modules import Scale

#-----------------------------------------------------------------
from yolox.nv_qdq import QDQ
#-----------------------------------------------------------------

EPS = 1E-4
PI_half = math.pi * 0.5


class OBBHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", 
        depthwise=False, with_scale=False, use_trig_loss=False, loss_weight_dict=None,
        quantize:bool = False):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        loss_weight_dict = dict(obj_loss_weight=1.0,
                                cls_loss_weight=1.0,
                                iou_loss_weight=5.0,
                                trig_loss_weight=1.0,
                                reg_loss_weight=1.0)
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        # self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.angle_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act, quantize=quantize ))  # 针对每个输入阶段
            # self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, ),
            #                                       Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, ), ]))  # 分类分支
            self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, quantize=quantize),
                                                  Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, quantize=quantize)]))  # 回归分支
            # self.cls_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * self.num_classes, kernel_size=1, stride=1, padding=0))  # 分类预测头
            if quantize:
              self.reg_preds.append(QDQ.quant_nn.QuantConv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 4, kernel_size=1, stride=1, padding=0))  # 回归预测头
              self.obj_preds.append(QDQ.quant_nn.QuantConv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0))  # 目标预测头
              self.angle_preds.append(QDQ.quant_nn.QuantConv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0))  # 角度预测头
            else:
              self.reg_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 4, kernel_size=1, stride=1, padding=0))  # 回归预测头
              self.obj_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0))  # 目标预测头
              self.angle_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0))  # 角度预测头

        # loss weight
        self.use_reg_loss = True
        self.use_trig_loss = True

        self.with_scale = with_scale
        self.reg_loss = L1Loss(reduction='none', loss_weight=loss_weight_dict["reg_loss_weight"])
        self.obj_loss = CELoss(reduction="none", loss_weight=loss_weight_dict["obj_loss_weight"])
        # self.cls_loss = CELoss(reduction="none", loss_weight=loss_weight_dict["cls_loss_weight"])
        # self.iou_loss = PolyIOULoss(mode='linear', reduction="none", loss_weight=loss_weight_dict["iou_loss_weight"])
        self.iou_loss = KLDloss()

        if self.use_trig_loss:
            self.trig_loss = L1Loss(reduction='none', loss_weight=loss_weight_dict["trig_loss_weight"])

        self.hw=[[80, 80], [40, 40], [20, 20]]

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        # self.obb_overlaps = OBBOverlaps()
        if self.with_scale:
            self.scale_trig = Scale(value=1.)
            self.scale_reg = Scale(value=1.)
        
        ## lixxxx  here we not exec quantize
        self.PI_half = torch.ones(1,1,1,1)  * PI_half 
        self.myconv = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.myconv.weight = torch.nn.Parameter(self.PI_half, requires_grad=False)

    def initialize_biases(self, prior_prob):
        # for conv in self.cls_preds:
        #     b = conv.bias.view(self.n_anchors, -1)
        #     b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        #     conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.angle_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None):
        outputs = []
        origin_reg_preds = []
        origin_angle_preds = []
        xy_shifts = []
        expanded_strides = []

        # for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
        for k, (reg_conv, stride_this_level, x) in enumerate(zip(self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            # cls_x = x
            reg_x = x

            # cls_feat = cls_conv(cls_x)
            # cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            angle_output = self.angle_preds[k](reg_feat)

            if self.with_scale:
                reg_output = self.scale_reg(reg_output)
                angle_output = self.scale_trig(angle_output)

            ############### for lixxxxyyyy
            #angle_output = angle_output.sigmoid() * PI_half

            ############################### way 1
            #angle_output = F.conv2d(angle_output.sigmoid(), weight = self.PI_half, bias=None, stride=1, padding=0)
            ############################### way 2
            # with torch.no_grad():
            #   angle_output = self.myconv(angle_output.sigmoid())
            ################################

            ############### for deploy dla  lixxxxyyyy
            angle_output = angle_output.sigmoid()

            if self.training:
                # output = torch.cat([reg_output, angle_output, obj_output, cls_output], 1)
                output = torch.cat([reg_output, angle_output, obj_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                xy_shifts.append(grid)
                expanded_strides.append(xin[0].new_full((1, grid.shape[1]), stride_this_level))
                if self.use_reg_loss or self.use_trig_loss:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                if self.use_reg_loss:
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).view(batch_size, -1, 4)
                    origin_reg_preds.append(reg_output.clone())
                if self.use_trig_loss:
                    angle_output = angle_output.view(batch_size, self.n_anchors, 1, hsize, wsize)
                    angle_output = angle_output.permute(0, 1, 3, 4, 2).view(batch_size, -1, 1)
                    origin_angle_preds.append(angle_output.clone())

                # logger.info('use_reg_loss: {}'.format(self.use_reg_loss))
                # logger.info('use_trig_loss: {}'.format(self.use_trig_loss))

            else:
                # output = torch.cat([reg_output, angle_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                output = torch.cat([reg_output, angle_output, obj_output.sigmoid()], 1)

            outputs.append(output)  # 包含三个stages的输出

        if self.training:
            return self.get_losses(xy_shifts, expanded_strides, labels, torch.cat(outputs, 1), origin_reg_preds, origin_angle_preds, dtype=xin[0].dtype)
        else:
            ############### for deploy dla  lixxxxyyyy
            return outputs
            self.hw = [x.shape[-2:] for x in outputs]

            # [batch, n_anchors_all, 86]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # shape(bs, n_anchor * h * w, 86)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        # n_ch = 6 + self.num_classes
        n_ch = 6
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []

        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)  # shape(1, w * h, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))  # shape(1, w * h, 1)

        grids = torch.cat(grids, dim=1).type(dtype)  # shape(1, sigma(w*h), 2)
        strides = torch.cat(strides, dim=1).type(dtype)  # same as above

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def decode_outputs_trt(self, outputs, dtype):
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # shape(bs, n_anchor * h * w, 86)
        outputs = self.decode_outputs(outputs, dtype)
        outputs[..., 4] = outputs[..., 4] * math.pi / 2
        return outputs

    def get_losses(self, xy_shifts, expanded_strides, labels, outputs, origin_reg_preds, origin_angle_preds, dtype):
        reg_preds = outputs[:, :, :5]  # [batch, n_anchors_all, 5]   归一
        obj_preds = outputs[:, :, 5].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        # cls_preds = outputs[:, :, 6:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        xy_shifts = torch.cat(xy_shifts, 1)  # shape(1, n_anchors_all, 2)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_reg_loss:
            origin_reg_preds = torch.cat(origin_reg_preds, 1)  # shape(bs, n_anchors_all, 4)
        if self.use_trig_loss:
            origin_angle_preds = torch.cat(origin_angle_preds, 1)

        # cls_targets = []
        reg_targets = []
        angle_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                # cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 5))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                if self.use_reg_loss:
                    l1_target = outputs.new_zeros((0, 4))
                if self.use_trig_loss:
                    angle_target = outputs.new_zeros((0, 1))
            else:
                gt_rbboxes_per_image = labels[batch_idx, :num_gt, 1:6]
                gt_classes = labels[batch_idx, :num_gt, 0]
                reg_preds_per_image = reg_preds[batch_idx]
                # try:
                (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = self.get_assignments(batch_idx,
                                                                                                                           num_gt, total_num_anchors, gt_rbboxes_per_image, gt_classes,
                                                                                                                           reg_preds_per_image, expanded_strides, xy_shifts, obj_preds)

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)  # shape(num_pos, num_classes)
                obj_target = fg_mask.unsqueeze(-1)  # shape(n_anchors_all, 1)
                reg_target = gt_rbboxes_per_image[matched_gt_inds]  # shape(num_pos, 5)

                if self.use_reg_loss:
                    l1_target = self.get_reg_l1_target(outputs.new_zeros((num_fg_img, 4)), reg_target, expanded_strides[0][fg_mask].unsqueeze(-1), xy_shifts=xy_shifts[0][fg_mask])
                if self.use_trig_loss:
                    angle_target = gt_rbboxes_per_image[matched_gt_inds][..., 4]
                    angle_target = self.get_angle_l1_target(outputs.new_zeros((num_fg_img, 1)), angle_target)

            # cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_reg_loss:
                l1_targets.append(l1_target)
            if self.use_trig_loss:
                angle_targets.append(angle_target)
        reg_targets = torch.cat(reg_targets, 0)
        # cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        if self.use_reg_loss:
            l1_targets = torch.cat(l1_targets, 0)
        if self.use_trig_loss:
            angle_targets = torch.cat(angle_targets, 0)

        # todo: check obj loss
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(reg_preds.view(-1, 5)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.obj_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        # loss_cls = (self.cls_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg

        if self.use_reg_loss:
            loss_reg = (self.reg_loss(origin_reg_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_reg = 0.0
        if self.use_trig_loss:
            loss_trig = (self.trig_loss(origin_angle_preds.view(-1, 1)[fg_masks], angle_targets)).sum() / num_fg
        else:
            loss_trig = 0.0

        # loss = loss_iou + loss_obj + loss_cls + loss_reg + loss_trig
        loss = loss_iou + loss_obj + loss_reg + loss_trig
        outputs_dict = {"total_loss": loss,
                        "loss_iou": loss_iou,
                        "loss_obj": loss_obj,
                        # "loss_cls": loss_cls,
                        "num_fg": num_fg / max(num_gts, 1)}

        if self.use_reg_loss:
            outputs_dict.update({"loss_reg": loss_reg})
        if self.use_trig_loss:
            outputs_dict.update({"loss_trig": loss_trig})

        return outputs_dict

    def get_reg_l1_target(self, l1_target, gt, stride, xy_shifts, eps=1e-8):
        l1_target[:, :2] = gt[:, :2] / stride - xy_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride.squeeze(-1) + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride.squeeze(-1) + eps)
        return l1_target

    def get_angle_l1_target(self, l1_target, gt):
        l1_target[..., 0] = gt
        return l1_target

    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, total_num_anchors, gt_rbboxes_per_image, gt_classes, reg_preds_per_image, expanded_strides, xy_shifts, obj_preds):

        # gt_bboxes_per_image:shape(n_gt, 5) bboxes_preds_per_image:shape(a_anchors, 5)
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_rbboxes_per_image, expanded_strides, xy_shifts, total_num_anchors, num_gt)
        assert fg_mask.any()
        reg_preds_per_image = reg_preds_per_image[fg_mask]  # shape(num_in, 5)
        # cls_preds_ = cls_preds[batch_idx][fg_mask]  # shape(num_in, num_classes)
        obj_preds_ = obj_preds[batch_idx][fg_mask]  # shape(num_in, 1)
        num_in_boxes_anchor = reg_preds_per_image.shape[0]

        gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1))  # shape(num_gt, num_in, 1)
        # pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        pair_wise_ious_loss = compute_kld_loss(gt_rbboxes_per_image, reg_preds_per_image)  # add
        pair_wise_ious = 1.0 - pair_wise_ious_loss

        with torch.cuda.amp.autocast(enabled=False):
            # cls_preds_ = (cls_preds_.float().unsqueeze(0).sigmoid_() * obj_preds_.float().unsqueeze(0).sigmoid_()).repeat(num_gt, 1, 1)
            # pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
            # del cls_preds_
            obj_preds_ = obj_preds_.float().unsqueeze(0).sigmoid_().repeat(num_gt, 1, 1)
            pair_wise_cls_loss = F.binary_cross_entropy(obj_preds_, gt_cls_per_image, reduction="none").sum(-1)
        del obj_preds_

        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center))

        (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg,

    def get_in_boxes_info(self, gt_rbboxes_per_image, expanded_strides, xy_shifts, total_num_anchors, num_gt):
        gt_angles_per_image = gt_rbboxes_per_image[:, 4, None]  # shape(num_gt, 1)
        gt_xy_per_image = gt_rbboxes_per_image[:, None, 0:2]  # shape(num_gt, 1, 2)
        gt_wh_per_image = gt_rbboxes_per_image[:, None, 2:4]  # shape(num_gt, 1, 2)
        expanded_strides_per_image = expanded_strides[..., None]
        xy_shifts_per_image = xy_shifts * expanded_strides_per_image
        grid_xy_per_image = xy_shifts_per_image + 0.5 * expanded_strides_per_image  # shape(1, n_anchor, 2)

        # in box
        Cos, Sin = torch.cos(gt_angles_per_image), torch.sin(gt_angles_per_image)  # shape(num_gt, 1)
        # Matric = torch.stack([Cos, -Sin, Sin, Cos], dim=-1).repeat(1, total_num_anchors, 1, 1).view(num_gt, total_num_anchors, 2, 2)
        Matric = torch.stack([Cos, Sin, -Sin, Cos], dim=-1).repeat(1, total_num_anchors, 1, 1).view(num_gt, total_num_anchors, 2, 2)
        offset = (grid_xy_per_image - gt_xy_per_image)[..., None]  # shape(num_gt, n_anchor, 2, 1)
        offset = torch.matmul(Matric, offset).squeeze_(-1)  # shape(n_gt, n_anchors, 2)

        b_lt = gt_wh_per_image / 2 + offset
        b_rb = gt_wh_per_image / 2 - offset
        bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)  # shape(n_gt, n_anchors, 4)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # shape(n_gt, n_anchors)
        is_in_boxes_all = is_in_boxes.sum(0) > 0  # shape(n_anchors)
        # in center
        center_radius = 2.5  # TODO: 2.5 -> 3.5
        c_dist = center_radius * expanded_strides_per_image  # shape(1, n_anchors_all, 1)
        c_lt = grid_xy_per_image - (gt_xy_per_image - c_dist)
        c_rb = (gt_xy_per_image + c_dist) - grid_xy_per_image

        center_deltas = torch.cat([c_lt, c_rb], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # shape(num_gts, n_anchors_all)
        is_in_centers_all = is_in_centers.sum(dim=0) > 0  # shape(n_anchors_all)

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # 在bboxes和centers内的全部取出
        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor])
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)  # shape(num_gt, num_in)
        ious_in_boxes_matrix = pair_wise_ious  # shape(num_gt, num_in)
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))

        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx, ks

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # shape(num_pos) 在所有的anchor中取出fg_mask，然后在fg_mask中取出pos

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
