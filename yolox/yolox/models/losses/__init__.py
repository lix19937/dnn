# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 下午3:44
# @Author  : Teanna
# @File    : __init__.py.py
# @Software: PyCharm
from .losses import L1Loss, CELoss, IOUloss
from .focal_loss import FocalLoss
from .rotated_iou_loss import RotatedIoULoss
from .cross_entropy_loss import CrossEntropyLoss
from .KLD_loss import KLDloss, compute_kld_loss
