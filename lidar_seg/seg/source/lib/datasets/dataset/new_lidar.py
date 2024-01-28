from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import os.path as osp
import numpy as np


class NewLidar:
    CLASSES = ('路面', '人行道', '泥地', '护栏', '墙体', '建筑', '高空物', '支撑杆', '低空植被',
       '交通牌', '交通锥', '其他目标', '车或者人', '轮挡', '关闸', '反射噪声',
       '水雾沙尘', '挡板', '红绿灯', '未知目标',)

    PALETTE = [[255, 0, 0], [255, 127, 80], [218, 97, 17],
           [255, 255, 0], [0, 200, 200], [128, 128, 0], [200, 200, 200],
           [107, 142, 35], [0, 255, 127], [152, 251, 152], [100, 60, 255], [142, 0, 252],
           [0, 0, 255], [60, 179, 113], [255, 215, 0], [0, 60, 100],
           [0, 80, 100], [72, 51, 159], [119, 11, 32], [128, 64, 128]]

    def __init__(self, opt, split):
        workspace_folder = '/'.join(osp.abspath(__file__).split('/')[:-5])
        data_root = osp.join(workspace_folder, 'data', 'seg_pre')
        data_dir = osp.join(data_root, 'lidar')
        self.images = os.listdir(data_dir)
        self.images = [osp.join(data_dir, _) for _ in self.images]
        self.img_suffix = '.pcd'
        self.seg_map_suffix = '.bin'
        self.split = split
        self.test_mode = split == 'val'
        self.labels = [_.replace('/lidar/', '/label_bin/')[:-4] + '.bin' for _ in self.images]
        self.calibrations = [_.replace('/lidar/', '/calibration/')[:-4] + '.yaml' for _ in self.images]
        self.opt = opt

        if self.test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` should be specified when testing'

        print(f'Loaded {len(self.images)} {split} images')

    def __len__(self):
        """Total number of samples of data."""
        return len(self.images)
