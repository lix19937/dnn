#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np
from loguru import logger
# from yolox.utils import xyxy2cxcywh, ploy2obb
from yolox.data.transforms import poly2obb_np_batch


def get_affine_matrix(target_size, degrees=10, translate=0.1, scales=0.1, shear=10, ):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * num_gts, 2)  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = np.concatenate((corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))).reshape(4, num_gts).T

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(img,
                  targets=(),
                  target_size=(640, 640),
                  degrees=10,
                  translate=0.1,
                  scales=0.1,
                  shear=10,
                  ):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5):
    height, width = image.shape[:2]
    if random.random() < prob:
        if random.random() < 0.5:
            image = image[:, ::-1]
            boxes[:, 0::2] = width - boxes[:, 0::2]
        else:
            image = image[::-1, :]
            boxes[:, 1::2] = width - boxes[:, 1::2]

    return image, boxes

# 'mean': [127, 107, 153], 'std': [56.5115, 29.692, 49.4464]
def preproc(img, input_size, swap=(2, 0, 1), mean=None, std=None):
    padded_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    #print("input_size",input_size, img.shape)  (640, 640, 3) #hwc

    # norm
    if mean is not None and std is not None:
        padded_img -= np.array(mean).reshape(1, 1, 3)
        padded_img /= np.array(std).reshape(1, 1, 3)
        # padded_img /= 255.0

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, angle_version='oc', mean_std=None):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.angle_version = angle_version

        self.mean = mean_std['mean'] if mean_std else None
        self.std = mean_std['std'] if mean_std else None

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, r_o = preproc(image, input_dim, mean=self.mean, std=self.std)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        boxes_o = poly2obb_np_batch(boxes_o, self.angle_version)  # bbox_o: [xyxy] to [c_x,c_y,w,h]

        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, mean=self.mean, std=self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = poly2obb_np_batch(boxes, self.angle_version)
        boxes[..., :-1] *= r_
        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, mean=self.mean, std=self.std)
            boxes_o[..., :-1] *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 6))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

        return image_t, padded_labels


class ValTransform:
    """
    only for demo !!!

    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, mean_std=None):
        self.mean = mean_std['mean'] if mean_std else None
        self.std = mean_std['std'] if mean_std else None

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, mean=self.mean, std=self.std)
        return img, np.zeros((1, 5))
