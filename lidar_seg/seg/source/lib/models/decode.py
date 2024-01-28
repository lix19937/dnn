from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from .utils import _transpose_and_gather_feat
from opts import opts


opt = opts().parse()


def _rot_alpha(rot):
    return np.arctan2(rot[:, 0], rot[:, 1])


def lidar_post(dets):
    roi_x_min = opt.roi[0]
    roi_y_min = opt.roi[1]
    bev_res = opt.bev_res
    down_ratio = opt.down_ratio

    det_re = np.zeros((dets.shape[0], 9))
    det_re[:, 0] = dets[:, 0] * down_ratio * bev_res + roi_x_min  # x
    det_re[:, 1] = dets[:, 1] * down_ratio * bev_res + roi_y_min   # y
    det_re[:, 2] = dets[:, 2]                   # z
    det_re[:, 3] = np.exp(dets[:, 3])    # l
    det_re[:, 4] = np.exp(dets[:, 4])  # w
    det_re[:, 5] = np.exp(dets[:, 5])  # h
    det_re[:, 6] = _rot_alpha(dets[:, 6:8])  # heading
    det_re[:, 7] = dets[:, 8]            # score
    det_re[:, 8] = dets[:, 9]  # cls
    return det_re


def _rot_alpha_gpu(rot):
    return torch.arctan(rot[:, 0] / (rot[:, 1] + 1e-10))
    # opset version 11 doens't support atan2.
    # return torch.atan2(rot[:, 0], rot[:, 1])


def lidar_post_gpu(dets):
    roi_x_min = opt.roi[0]
    roi_y_min = opt.roi[1]
    bev_res = opt.bev_res
    down_ratio = opt.down_ratio

    det_re = torch.zeros([dets.shape[0], 9], device='cuda')
    det_re[:, 0] = dets[:, 0] * down_ratio * bev_res + roi_x_min  # x
    det_re[:, 1] = dets[:, 1] * down_ratio * bev_res + roi_y_min   # y
    det_re[:, 2] = dets[:, 2]                   # z
    det_re[:, 3] = torch.exp(dets[:, 3])    # l
    det_re[:, 4] = torch.exp(dets[:, 4])  # w
    det_re[:, 5] = torch.exp(dets[:, 5])  # h
    det_re[:, 6] = _rot_alpha_gpu(dets[:, 6:8])  # heading
    det_re[:, 7] = dets[:, 8]            # score
    det_re[:, 8] = dets[:, 9]  # cls
    return det_re


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _score_greater_than(scores, thr=0.5):
    batch, cat, height, width = scores.size()
    # Attention: score, cls, and idx must be corresponding.
    out_pos = scores.gt(thr)
    topk_score = torch.masked_select(scores, out_pos)

    out_pos = torch.nonzero((out_pos == True), as_tuple=False)
    topk_inds = out_pos[:, 2] * width + out_pos[:, 3]
    topk_inds = topk_inds.unsqueeze(0)   # The batch is 1.
    # topk_score, _ = torch.topk(scores.view(batch, -1), topk_inds.shape[1])

    topk_clses = out_pos[:, 1]
    topk_clses = topk_clses.unsqueeze(0)  # The batch is 1.

    topk_inds = topk_inds % (height * width)
    topk_xs = (topk_inds / width).int().float()
    topk_ys = (topk_inds % width).int().float()

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _top_k(scores, K=128):
    batch, cat, height, width = scores.size()
    # N x K.
    topk_scores, topk_inds = torch.topk(
        scores.view(batch, -1), K, dim=-1, largest=True, sorted=True)
    # inds = c0 * H*W + h0 * W + w0.
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_xs = (topk_inds / width).int().float()
    topk_ys = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def lidar_od_decode(heat, rot, center_z, dim, reg=None, top_k=128):
    batch, cat, height, width = heat.size()

    heat = _nms(heat)
    scores, inds, clses, ys, xs = _top_k(heat, top_k)

    K = top_k
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 2)
    z = _transpose_and_gather_feat(center_z, inds)
    z = z.view(batch, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    detections = torch.cat(
        [xs, ys, z, dim, rot, scores, clses], dim=2)  # B K 10
    return detections
