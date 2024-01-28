# -*- coding: utf-8 -*-
# @Time    : 2022/8/16 下午2:46
# @Author  : Teanna
# @File    : obb_decoder.py
# @Software: PyCharm

from loguru import logger

import torch
from .bbox_nms_rotated import multiclass_nms_rotated

def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)

# def obb2poly(obboxes):
#     center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
#     Cos, Sin = torch.cos(theta), torch.sin(theta)
#
#     vector1 = torch.cat([w / 2 * Cos, -w / 2 * Sin], dim=-1)
#     vector2 = torch.cat([-h / 2 * Sin, -h / 2 * Cos], dim=-1)
#
#     point1 = center + vector1 + vector2
#     point2 = center + vector1 - vector2
#     point3 = center - vector1 - vector2
#     point4 = center - vector1 + vector2
#     return torch.cat([point1, point2, point3, point4], dim=-1)


def obbpostprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    outputs = [None for _ in range(len(prediction))]
    for i, pred in enumerate(prediction):
        if not pred.size(0) > 0:
            continue
        rboxes = pred[:, :5]

        obj_conf = pred[:, 5].unsqueeze(-1)
        # cls_out = pred[:, 6: 6 + num_classes]
        # class_conf, class_pred = torch.max(cls_out, 1, keepdims=True)

        obj_conf = torch.cat([obj_conf, torch.zeros_like(obj_conf)], dim=-1)
        reserved, _, index = multiclass_nms_rotated(rboxes, obj_conf, conf_thre, nms_thre, return_inds=True)
        if not reserved.size(0):
            continue
        # logger.info('after: {}'.format(reserved))
        # logger.info('class: {}'.format(class_conf[index]))

        result_poly = obb2poly_oc(reserved[..., :5])
        result = torch.cat([result_poly, reserved[..., -1:], torch.ones_like(reserved[..., -1:]), torch.zeros_like(reserved[..., -1:])], dim=-1)

        if outputs[i] is None:
            outputs[i] = result
        else:
            outputs[i] = torch.cat([outputs[i], result])
        # logger.info('conf_thre:{}, nms_thre:{}'.format(conf_thre, nms_thre))
        # logger.info('result:{}'.format(result))
    return outputs



