from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .base_trainer import BaseTrainer
from .loss.lovasz_softmax import Lovasz_softmax
from .loss.boundary_loss import BoundaryLoss
from .loss.dice_loss import DiceLoss
from .loss.focal_loss import FocalLoss

class LidarSegLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.weight = torch.from_numpy(np.array([1,1,1,1,1,1,10,1,1,1], dtype=np.float32))
        self.ce = nn.CrossEntropyLoss(ignore_index=opt.ignore_index, reduction='none')
        self.ls = Lovasz_softmax(opt.ignore_index)
        self.bl = BoundaryLoss()
        self.dl = DiceLoss(ignore_index=opt.ignore_index, reduction='none')
        self.fl = FocalLoss(alpha=self.weight, ignore_index=opt.ignore_index)

    def forward(self, outputs, batch):
        # Uses only the last stack output.
        output = outputs[-1]['seg']
        loss = 1.4 * self.ce(output, batch['gt_segment_label'].squeeze(1).long()) + \
        0.6 * self.ls(F.softmax(output, dim=1), batch['gt_segment_label'].long())
        
        loss_state = {"loss": loss}
        return loss, loss_state


class LidarSegTrainer(BaseTrainer):
    def __init__(self, opt, model, local_rank, optimizer=None):
        super().__init__(opt, model, local_rank, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss']
        loss = LidarSegLoss(opt)
        return loss_states, loss
