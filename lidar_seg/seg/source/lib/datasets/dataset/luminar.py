from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import os.path as osp
import numpy as np


class Luminar:
    def __init__(self, opt, split):
        workspace_folder = '/'.join(osp.abspath(__file__).split('/')[:-5])
        data_root = osp.join(workspace_folder, 'data', 'seg', 'table')
        files = open(osp.join(data_root, split + '.txt')).readlines()
        self.images = []
        for file in files:
            data_path = osp.join(data_root, file.strip())
            data_dir = osp.dirname(data_path).replace('scene', 'lidar')
            data = open(data_path, 'r').readlines()
            self.images += [osp.join(data_dir, _.strip()) for _ in data]
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
