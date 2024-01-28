from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import os.path as osp
import numpy as np
from loguru import logger

class MiniData:
    CLASSES = ('路面', '人行道', '障碍物', '高空噪声', '支撑杆', '交通牌', '交通锥',
           '车或者人', '轮挡', '挡板', '未知目标',)

    PALETTE = [[255, 0, 0], [255, 127, 80], [0, 255, 127],
        [200, 200, 200], [107, 142, 35], [0, 255, 127],
        [152, 251, 152], [0, 0, 255], [142, 0, 252], [128, 64, 128]]

    def __init__(self, opt, split, is_calib = False):
        workspace_folder = '/'.join(osp.abspath(__file__).split('/')[:-5])
        data_root = osp.join(workspace_folder, 'data', 'seg', 'aligned_seg')
        # data_root = "/Data/lidar_data/seg-2.0w/aligned_seg"
        data_root = "/Data/luminar_seg/aligned_seg"
        if is_calib:
            data_root = "/Data/luminar_seg_calib/aligned_seg"
        
        data_file = os.path.join(data_root, split + '.txt')
        # filter_file = os.path.join(data_root, 'filter.txt')
        if os.path.exists(data_file):
            datas = open(data_file, 'r').readlines()
            self.images = []
            for task in datas:
                tmp = open(os.path.join(data_root, task.strip()), 'r').readlines()
                tmp = [_.strip() for _ in tmp]
                self.images = self.images + tmp
            self.labels = [_.replace('pcd_dir', 'ann_dir') for _ in self.images]
        else:
          print('not exist data_file >>>:', data_file);exit(0)

        # task_dir = [os.path.join(data_root, _) for _ in data_list if _ not in filters]
        self.test_mode = split == 'val'
        self.split = split
        self.opt = opt

        #print(f'Loaded {len(self.images)} {self.split} images')
        logger.info("phase:{}, loaded {} {} images".format("calib" if is_calib else "train", len(self.images), self.split ))

    def __len__(self):
        """Total number of samples of data."""
        return len(self.images)