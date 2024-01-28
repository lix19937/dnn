from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

import torch.utils.data as data
import numpy as np
import math
import os.path as osp
# from . import preprocess_cpp

lidar_seg_gt_map = {0:0, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:4, 8:2, 9:5, 10:6,
          11:2, 12:7, 13:8, 14:2, 15:3, 16:3, 17:10, 18:9, 19:3}

new_lidar_gt_map = {0:0, 1:1, 2:1, 3:0, 4:2, 5:2, 6:2, 7:3, 8:4, 9:2, 10:5, 11:6,
          12:2, 13:7, 14:8, 15:2, 16:3, 17:3, 18:11, 19:9, 20:10}

def align_cloud_size(cloud, max_size=50000, constant_value=0):
    if cloud.shape[1] >= max_size:
        return cloud[:, :max_size, ...]
    else:
        return np.pad(cloud, ((0, 0), (0, max_size - cloud.shape[1])), constant_values=constant_value)

def label_change(label):
    label[label > 19] = 17
    label = np.vectorize(lidar_seg_gt_map.get)(label).astype(np.uint8)
    return label

def new_label_change(label):
    label[label > 20] = 18
    label = np.vectorize(new_lidar_gt_map.get)(label).astype(np.uint8)
    return label

class LidarSeg(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.img_scale = (self.opt.input_c, self.opt.input_h, self.opt.input_w)
        self.transform = np.eye(3)
        self.transform[0, 2] = -self.opt.roi[0]
        self.transform[1, 2] = -self.opt.roi[1]
        self.transform = self.transform / self.opt.bev_res
        self.aligned_size = self.opt.align_size

    def _in_roi(self, index):
        return 0 <= index[0] < self.img_scale[1] and 0 <= index[1] < self.img_scale[2]

    def _augment(self, lidar):
        # translation = np.random.uniform(-1, 1, size=(3, 1)).astype(np.float32)
        euler = np.random.randint(-5, 5, size=3).astype(np.float) / 10
        euler[0] = 0
        from scipy.spatial.transform.rotation import Rotation as R
        rot = R.from_euler('zyx', euler, degrees=True).as_matrix().astype(np.float32)
        transformed_cloud = np.matmul(rot, lidar[:, :3].transpose()).astype(np.float32)

        # t = TransformLidar(translation, euler, self.seq)
        # results = t(results)
        return transformed_cloud.T


    def project_to_bev(self, cloud):
        """Resize images with ``results['scale']``."""
        ori_cloud = cloud.copy()
        ori_cloud[:, -1] = 1
        trans_cloud = np.matmul(self.transform, ori_cloud.T)
        indices = trans_cloud[:2, :].astype(np.int32).T
        mask = np.zeros((1, indices.shape[0]), np.int32)
        dict_mat = dict()
        for idx in range(indices.shape[0]):
            tmp_tuple = tuple(indices[idx, :])
            if tmp_tuple not in dict_mat:
                dict_mat[tmp_tuple] = [idx]
            else:
                dict_mat[tmp_tuple].append(idx)

        # img_ori = np.zeros(self.img_scale, 'f')
        for key, val in dict_mat.items():
            if self._in_roi(key):
                mask[0, val] = 1
        # Transforms cloud to bev feature map, which is input for backbone.
        # Python too slow. Uses cpp extension instead.
        lidar_one_frame = torch.from_numpy(cloud)
        inp = preprocess_cpp.build(
            lidar_one_frame,
            self.opt.channels_num,
            self.opt.input_h,
            self.opt.input_w,
            self.opt.roi[0],
            self.opt.roi[1],
            self.opt.roi[2],
            self.opt.roi[5],
            self.opt.bev_res,
            self.opt.height_res)
        inp = inp.numpy()
        inp[0, :, :] /= self.opt.roi[3]
        inp[1, :, :] /= self.opt.roi[4]
        inp[2:, :, :] = np.log(inp[2:, :, :] + 1.0)

        indices = align_cloud_size((indices[:, 0] * self.img_scale[-1] + indices[:, 1]).reshape(1, -1),
                                   self.aligned_size)
        mask = align_cloud_size(mask, self.aligned_size)
        results = dict()
        results['input'] = inp
        results['aligned_cloud'] = align_cloud_size(cloud.transpose(), self.aligned_size)
        results['indices'] = indices
        results['mask'] = mask
        return results

    def __getitem__(self, index):
        # Reads labels and calibration.
        seg_label_path = self.labels[index]
        gt_seg_label = np.fromfile(seg_label_path, np.uint8).reshape(1, -1)
        label_change(gt_seg_label)

        # Reads lidar data.
        data_path = self.images[index]
        lidar_one_frame = np.fromfile(data_path, np.float32).reshape(-1, 3)

        # Augmentation.
        aug_prob = np.random.uniform(0.0, 1.0)
        if self.split == 'train' and aug_prob < self.opt.aug_lidar:
            lidar_one_frame = self._augment(lidar_one_frame)

        ret = self.project_to_bev(lidar_one_frame)
        ret["gt_segment_label"] = align_cloud_size(gt_seg_label, self.aligned_size, self.opt.ignore_index)
        ret["gt_segment_label"][ret['mask'] == 0] = self.opt.ignore_index
        ret["meta"] = data_path
        return ret
