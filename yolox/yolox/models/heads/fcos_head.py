# -*- coding: utf-8 -*-
# @Time    : 2022/8/2 下午12:10
# @Author  : Teanna
# @File    : fcos_head.py
# @Software: PyCharm
import math
import os

from loguru import logger
import torch
from torch import nn
from mmcv.runner import force_fp32

from yolox.models.components.network_blocks import BaseConv
from yolox.models.losses import RotatedIoULoss, FocalLoss, CrossEntropyLoss
from yolox.utils import DistanceAnglePointCoder, DistancePointBBoxCoder, LayersPointGenerator, multi_apply, reduce_mean
from yolox.utils.post_processing import multiclass_nms_rotated

INF = 1e8


class Head(nn.Module):
    def __init__(self, num_classes=1, width=1.0, strides=None, in_channels=None, act='silu', angle_version='oc', demo_cfg=None):
        super().__init__()

        self.mode = os.getenv('MODE', 'train')

        if in_channels is None:
            in_channels = [256, 512, 1024]
        if strides is None:
            strides = [8, 16, 32]
        self.n_anchors = 1
        self.strides = strides
        self.center_sampling = True
        self.separate_angle = False  # todo: add l1 loss

        self.center_sample_radius = 1.5
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.prior_generator = LayersPointGenerator(self.strides)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.angle_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.center_preds = nn.ModuleList()

        self.stems = nn.ModuleList()
        Conv = BaseConv

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act, ))
            self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, ),
                                                  Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, ), ]))
            self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, ),
                                                  Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act, ), ]))

            # todo: kernel_size
            self.cls_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, ))
            self.reg_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=(1, 1), stride=(1, 1), padding=0, ))
            self.angle_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, ))
            # self.obj_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, ))
            self.center_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, ))

        self.loss_centerness = CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)
        self.loss_bbox = RotatedIoULoss()
        self.loss_cls = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)

        if self.separate_angle:
            self.h_bbox_coder = DistancePointBBoxCoder()
            self.loss_angle = nn.L1Loss(reduction="none")
        else:
            self.bbox_coder = DistanceAnglePointCoder(angle_version=angle_version)

        self.debug = bool(int(os.getenv('DEBUG', 0)))

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats, labels=None):
        scores, bbox_preds, angle_preds, centernesses = [], [], [], []
        for index, feat in enumerate(feats):
            x = self.stems[index](feat)
            reg_x = x
            reg_x = self.reg_convs[index](reg_x)
            reg_output = self.reg_preds[index](reg_x)
            angle_output = self.angle_preds[index](reg_x)
            centerness = self.center_preds[index](reg_x)

            cls_x = x
            cls_x = self.cls_convs[index](cls_x)
            score = self.cls_preds[index](cls_x)

            # post
            reg_output = reg_output.clamp(min=0)

            scores.append(score)
            bbox_preds.append(reg_output)
            angle_preds.append(angle_output)
            centernesses.append(centerness)

        if self.mode == 'demo':
            return self.get_bboxes(scores, bbox_preds, angle_preds, centernesses)
        # if labels is None:
        #     return scores, bbox_preds, angle_preds, centernesses

        loss = self.loss(scores, bbox_preds, angle_preds, centernesses, labels)

        return loss

    @force_fp32(apply_to=('scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self, scores, bbox_preds, angle_preds, centernesses, gt_bboxes):
        logger.info('gt_bboxes: {}'.format(gt_bboxes))
        # exit(0)
        assert len(scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in scores]

        if self.debug:
            logger.info('featmap_sizes: {}'.format(featmap_sizes))

        all_level_points = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

        labels, bbox_targets, angle_targets = self.get_targets(all_level_points, gt_bboxes)

        num_imgs = scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        # todo: num_classes
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, scores[0].size(-3)) for cls_score in scores]
        flatten_cls_scores = torch.cat(flatten_cls_scores)

        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)

        flatten_angle_preds = [angle_pred.permute(0, 2, 3, 1).reshape(-1, 1) for angle_pred in angle_preds]
        flatten_angle_preds = torch.cat(flatten_angle_preds)

        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_centerness = torch.cat(flatten_centerness)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels.long(), avg_factor=num_pos)
        # loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels.long())

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds], dim=-1)
                pos_bbox_targets = torch.cat([pos_bbox_targets, pos_angle_targets], dim=-1)

            pos_decoded_bbox_preds = bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds, weight=pos_centerness_targets, avg_factor=centerness_denorm)
            if self.separate_angle:
                loss_angle = self.loss_angle(pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.separate_angle:
                loss_angle = pos_angle_preds.sum()

        if self.separate_angle:
            total_loss = loss_cls + loss_bbox + loss_angle + loss_centerness
            return dict(total_loss=total_loss, loss_cls=loss_cls, loss_bbox=loss_bbox, loss_angle=loss_angle, loss_centerness=loss_centerness)
        else:
            total_loss = loss_cls + loss_bbox + loss_centerness
            return dict(total_loss=total_loss, loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)

    def get_targets(self, all_level_points, gt_bboxes):
        num_levels = len(all_level_points)

        # todo: remove
        # regress_ranges = ((-1, 64), (64, 128), (128, 256))
        # expanded_regress_ranges = [all_level_points[i].new_tensor(regress_ranges[i])[None].expand_as(all_level_points[i]) for i in range(3)]

        concat_points = torch.cat(all_level_points, dim=0)
        num_points = [center.size(0) for center in all_level_points]
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(self._get_target_single, gt_bboxes,
                                                                         points=concat_points, )
        # regress_ranges=torch.concat(expanded_regress_ranges, dim=0),
        # num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]
        angle_targets_list = [angle_targets.split(num_points, 0) for angle_targets in angle_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat([angle_targets[i] for angle_targets in angle_targets_list])
            bbox_targets = bbox_targets / self.strides[i]

            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_angle_targets

    def _get_target_single(self, gt_bboxes, points):  # , regress_ranges, num_points_per_lvl):
        """Compute regression, classification and angle targets for a single image."""
        real_gt_bboxes = torch.sum(gt_bboxes, dim=-1) > 0
        gt_bboxes = gt_bboxes[real_gt_bboxes]
        if self.debug:
            logger.info('real_gt_bboxes: {}'.format(real_gt_bboxes))
            logger.info('gt_bboxes: {}'.format(gt_bboxes))

        num_points = points.size(0)  # 8600
        num_gts = gt_bboxes.size(0)  # todo: in yolox, the label size is constant , e.g. 50
        gt_labels = gt_bboxes[..., 0]
        gt_bboxes = gt_bboxes[..., 1:]

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), gt_bboxes.new_zeros((num_points, 4)), gt_bboxes.new_zeros((num_points, 1))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)

        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])

        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # if self.center_sampling:
        #     # condition1: inside a `center bbox`
        #     radius = self.center_sample_radius
        #     stride = offset.new_zeros(offset.shape)
        #
        #     # project the points on current lvl back to the `original` sizes
        #     lvl_begin = 0
        #     for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
        #         lvl_end = lvl_begin + num_points_lvl
        #         stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
        #         lvl_begin = lvl_end
        #
        #     inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
        #     inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask, inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        # max_regress_distance = bbox_targets.max(-1)[0]
        # inside_regress_range = ((max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]

        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]

        return labels, bbox_targets, angle_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, angle_preds, centernesses, cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            angle_preds (list[Tensor]): Box angle for each scale level \
                with shape (N, num_points * 1, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the 6-th
                column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        result_list = []
        for img_id in range(len(cls_scores[0])):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            angle_pred_list = [angle_preds[i][img_id].detach() for i in range(num_levels)]
            centerness_pred_list = [centernesses[i][img_id].detach() for i in range(num_levels)]

            det_bboxes = self._get_bboxes_single(cls_score_list, bbox_pred_list, angle_pred_list, centerness_pred_list, mlvl_points, cfg)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, angle_preds, centernesses, mlvl_points, cfg):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        # cfg = self.test_cfg if cfg is None else cfg
        if cfg is None:
            cfg = dict(min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000)
        logger.info(cfg)

        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(cls_scores, bbox_preds, angle_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            logger.info('bbox_pred: {}'.format(bbox_pred))

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            # bboxes = self.bbox_coder.decode(points, bbox_pred, max_shape=img_shape)
            bboxes = self.bbox_coder.decode(points, bbox_pred)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)

        logger.info(mlvl_bboxes)
        # if rescale:
        #     scale_factor = mlvl_bboxes.new_tensor(scale_factor)
        #     mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        logger.info('mlvl_bboxes:{}'.format(mlvl_bboxes))
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes, mlvl_scores,
                                                        cfg['score_thr'], cfg['nms'], cfg['max_per_img'],
                                                        score_factors=mlvl_centerness)
        logger.info('det_bboxes: {}'.format(det_bboxes))
        return det_bboxes, det_labels

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centerness'))
    # def refine_bboxes(self, cls_scores, bbox_preds, angle_preds, centernesses):
    #     """This function will be used in S2ANet, whose num_anchors=1."""
    #     num_levels = len(cls_scores)
    #     assert num_levels == len(bbox_preds)
    #     num_imgs = cls_scores[0].size(0)
    #     for i in range(num_levels):
    #         assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)
    #
    #     # device = cls_scores[0].device
    #     featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    #     mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
    #                                                    bbox_preds[0].dtype,
    #                                                    bbox_preds[0].device)
    #     bboxes_list = [[] for _ in range(num_imgs)]
    #
    #     for lvl in range(num_levels):
    #         bbox_pred = bbox_preds[lvl]
    #         angle_pred = angle_preds[lvl]
    #         bbox_pred = bbox_pred.permute(0, 2, 3, 1)
    #         bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
    #         angle_pred = angle_pred.permute(0, 2, 3, 1)
    #         angle_pred = angle_pred.reshape(num_imgs, -1, 1)
    #         bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)
    #
    #         points = mlvl_points[lvl]
    #
    #         for img_id in range(num_imgs):
    #             bbox_pred_i = bbox_pred[img_id]
    #             decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
    #             bboxes_list[img_id].append(decode_bbox_i.detach())
    #
    #     return bboxes_list
