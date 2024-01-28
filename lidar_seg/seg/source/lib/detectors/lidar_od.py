from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from progress.bar import Bar
import time
import torch
import json

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import lidar_od_decode
from .base_detector import BaseDetector
from trains.lidar_od import LidarOdLoss
import numpy as np


def rot_alpha(rot):
    return np.arctan2(rot[:, 0], rot[:, 1])


def lidar_post(dets):
    roi_x_min = -30.0
    roi_y_min = -51.2
    bev_res = 0.2
    det_re = np.zeros((1, 100, 9))
    det_re[0, :, 0] = dets[0, :, 0] * 4. * bev_res + roi_x_min  # x
    det_re[0, :, 1] = dets[0, :, 1] * 4. * bev_res + roi_y_min   # y
    det_re[0, :, 2] = dets[0, :, 2]                   # z
    det_re[0, :, 3] = np.exp(dets[0, :, 3])    # l
    det_re[0, :, 4] = np.exp(dets[0, :, 4])  # w
    det_re[0, :, 5] = np.exp(dets[0, :, 5])  # h
    det_re[0, :, 6] = rot_alpha(dets[0, :, 6:8])  # heading
    det_re[0, :, 7] = dets[0, :, 8]            # score
    det_re[0, :, 8] = dets[0, :, 9]  # cls
    return det_re


class LidarOdDetector(BaseDetector):

    def __init__(self, opt):
        super(LidarOdDetector, self).__init__(opt)
        self.loss_stats, self.loss = self._get_losses(opt)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'dim_loss',
                       'rot_loss', 'off_loss', 'height_loss']
        loss = LidarOdLoss(opt)
        return loss_states, loss

    def post_process(self, dets, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = lidar_post(
            dets.copy())
        return dets[0]

    def run(self, batch, opt, out_name):
        with torch.no_grad():
            outputs = self.model(batch['input'])
            output = outputs[-1]
            loss, loss_stats = self.loss(outputs, batch)
            hm = output['hm'].sigmoid_()
            rot = output['rot']
            z = output['z']
            dim = output['dim']
            reg = output['reg'] if self.opt.reg_offset else None
            torch.cuda.synchronize()
            # B K 10 [xs, ys, z, dim, rot, scores, clses]
            dets = lidar_od_decode(hm, rot, z, dim, reg, K=100)

        dets = self.post_process(dets, 1)
        print('dets.shape:', dets.shape)
        a = [value for value in dets.values() if len(value) > 0]
        b = np.vstack(a)
        c = np.where(b[:, 7] > 0.55)
        d = b[c]
        filename = 'out_result/'+out_name + '_result.json'
        with open(filename, 'w') as file_obj:
            json.dump(d.tolist(), file_obj)
        print('a frame is done')
        return filename
