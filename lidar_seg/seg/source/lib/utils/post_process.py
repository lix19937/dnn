from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .check_overlap import CheckOverlap

import time


def generate_one_box(obj):
    box = np.zeros((4, 2), dtype=np.float32)
    l = obj[3]
    w = obj[4]
    box[0] = np.array([l/2., w/2.], dtype=np.float32)
    box[1] = np.array([l/2., -w/2.], dtype=np.float32)
    box[2] = np.array([-l/2., -w/2.], dtype=np.float32)
    box[3] = np.array([-l/2., w/2.], dtype=np.float32)
    x_c = obj[0]
    y_c = obj[1]
    theta = obj[6]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    for idx in range(len(box)):
        x_tmp = cos_t * box[idx][0] - \
            sin_t * box[idx][1] + x_c
        y_tmp = sin_t * box[idx][0] + \
            cos_t * box[idx][1] + y_c
        box[idx][0] = x_tmp
        box[idx][1] = y_tmp
    return box


def generate_boxes(dets_post):
    box_list = []
    for obj in dets_post:
        box_list.append(generate_one_box(obj))
    return box_list


def is_type_head(cls):
    return cls > 3.5 and cls < 4.5


def is_type_truck(cls):
    return cls > 2.5 and cls < 3.5


def bev_nms(dets_post, socre_thr=0.5):
    overlap_checker = CheckOverlap()
    origin_boxes = generate_boxes(dets_post)
    for i in range(0, len(dets_post) - 1):
        score1 = dets_post[i][-2]
        if score1 < 0.0:
            continue
        if score1 > 0.0 and score1 < socre_thr:
            return dets_post
        box1 = origin_boxes[i]
        cls1 = dets_post[i][-1]
        for j in range(i + 1, len(dets_post)):
            score2 = dets_post[j][-2]
            if score2 < 0.0:
                continue
            if score2 > 0.0 and score2 < socre_thr:
                break
            box2 = origin_boxes[j]
            overlap = overlap_checker.is_overlap(box1, box2)
            if not overlap:
                continue
            cls2 = dets_post[j][-1]
            if (is_type_head(cls1) and is_type_truck(cls2)) or (is_type_head(cls2) and is_type_truck(cls1)):
                continue
            dets_post[j][-2] = -1.0
    return dets_post
