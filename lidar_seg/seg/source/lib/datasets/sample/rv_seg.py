from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

import torch.utils.data as data
import numpy as np
import math
import os.path as osp
from scipy.spatial.transform import Rotation as R
from .lidar_seg import label_change, new_label_change



class RvSeg(data.Dataset):
    CLASSES = ('路面', '人行道', '障碍物', '高空噪声', '支撑杆', '交通牌', '交通锥',
           '车或者人', '轮挡', '挡板', '红绿灯', '未知目标',)

    PALETTE = [[255, 0, 0], [255, 127, 80], [0, 255, 127],
        [200, 200, 200], [107, 142, 35], [0, 255, 127],
        [152, 251, 152], [0, 0, 255], [142, 0, 252], [119, 11, 32], [128, 64, 128]]
    # CLASSES = ('路面', '人行道', '泥地', '护栏', '墙体', '建筑', '高空物', '支撑杆', '低空植被',
    #    '交通牌', '交通锥', '其他目标', '车或者人', '轮挡', '关闸', '反射噪声',
    #    '水雾沙尘', '挡板', '红绿灯', '未知目标',)

    # PALETTE = [[255, 0, 0], [255, 127, 80], [218, 97, 17],
    #        [255, 255, 0], [0, 200, 200], [128, 128, 0], [200, 200, 200],
    #        [107, 142, 35], [0, 255, 127], [152, 251, 152], [100, 60, 255], [142, 0, 252],
    #        [0, 0, 255], [60, 179, 113], [255, 215, 0], [0, 60, 100],
    #        [0, 80, 100], [72, 51, 159], [119, 11, 32], [128, 64, 128]]

    def __init__(self) -> None:
        super().__init__()
        self.img_scale=(5, 192, 1024)
        fov_angle=(-16, 8)
        horizon_angle=(-64, 64)
        fov_up = fov_angle[1] / 180.0 * np.pi  # field of view up in rad
        self.fov_down = fov_angle[0] / 180.0 * np.pi  # field of view down in rad
        self.fov = fov_up - self.fov_down  # get field of view total in rad
        self.fov_left = horizon_angle[0] / 180.0 * np.pi
        self.horizon_fov = horizon_angle[1] / 180.0 * np.pi - self.fov_left

        # quat_xyzw = [-0.0061, -0.01, 0.002759, 1]
        # self.translation = np.array([1.633116386017969, 0.023100567710573547, 1.5548923643842074], np.float32)
        quat_xyzw = [0, 0, 0, 1]
        self.translation = np.array([0, 0, 2.0], np.float32)
        self.rotate = R.from_quat(quat_xyzw).as_matrix()

        # self.aligned_size = self.opt.align_size

    def _in_roi(self, index):
        return 0 <= index[0] < self.img_scale[1] and 0 <= index[1] < self.img_scale[2]

    def _augment(self, lidar):
        # translation = np.random.uniform(-1, 1, size=(3, 1)).astype(np.float32)
        euler = np.random.randint(-10, 10, size=3).astype(np.float) / 100
        euler[0] = 0
        from scipy.spatial.transform.rotation import Rotation as R
        rot = R.from_euler('zyx', euler, degrees=True).as_matrix().astype(np.float32)
        transformed_cloud = np.matmul(rot, lidar[:, :3].transpose()).astype(np.float32)

        # t = TransformLidar(translation, euler, self.seq)
        # results = t(results)
        return transformed_cloud.T


    def project_to_rv(self, ori_cloud, label):
        """Resize images with ``results['scale']``."""
        # ori_cloud = cloud.copy()
        # ori_cloud[:, -1] = 1
        cloud = (np.matmul(self.rotate.T, ori_cloud.T) - self.translation.reshape(3, -1)).T.astype(np.float32)
        cloud_size = cloud.shape[0]
        scan_x = cloud[:, 0]
        scan_y = cloud[:, 1]
        scan_z = cloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        depth = np.linalg.norm(cloud, 2, axis=1)
        pitch = np.arcsin(scan_z / depth)
        c = ((yaw - self.fov_left) / self.horizon_fov * self.img_scale[2]).astype(np.int32)  # in [0.0, 1.0]
        r = (self.img_scale[1] - (pitch - self.fov_down) / self.fov * self.img_scale[1]).astype(np.int32)  # in [0.0, 1.0]
        
        # mask = np.ones((1, cloud_size), np.int32)
        dict_mat = dict()
        for idx in range(cloud_size):
            tmp_tuple = (r[idx], c[idx])
            if tmp_tuple not in dict_mat:
                dict_mat[tmp_tuple] = [idx]
            else:
                dict_mat[tmp_tuple].append(idx)

        img_ori = np.zeros(self.img_scale, 'f')
        label_img = np.ones(self.img_scale[1:], np.uint8) * self.opt.ignore_index
        for key, val in dict_mat.items():
            if self._in_roi(key):
                img_ori[0, key[0], key[1]] = ori_cloud[:, 0][val[-1]]
                img_ori[1, key[0], key[1]] = ori_cloud[:, 1][val[-1]]
                img_ori[2, key[0], key[1]] = ori_cloud[:, 2][val[-1]]
                img_ori[3, key[0], key[1]] = len(val)
                img_ori[4, key[0], key[1]] = depth[val[-1]]
                label_img[key[0], key[1]] = label[0, val[-1]]
                
        results = dict()
        results['input'] = img_ori
        # results['aligned_cloud'] = cloud
        results['gt_segment_label'] = label_img
        # results['mask'] = mask
        return results

    def __getitem__(self, index):
        # Reads labels and calibration.
        seg_label_path = self.labels[index]
        gt_seg_label = np.fromfile(seg_label_path, np.uint8).reshape(1, -1)
        gt_seg_label = label_change(gt_seg_label)

        # Reads lidar data.
        data_path = self.images[index]
        lidar_one_frame = np.fromfile(data_path, np.float32).reshape(-1, 3)

        # Augmentation.
        aug_prob = np.random.uniform(0.0, 1.0)
        if self.split == 'train' and aug_prob < self.opt.aug_lidar:
            lidar_one_frame = self._augment(lidar_one_frame)

        ret = self.project_to_rv(lidar_one_frame, gt_seg_label)
        # ret["gt_segment_label"] = align_cloud_size(gt_seg_label, self.aligned_size, self.opt.ignore_index)
        # ret["gt_segment_label"][ret['mask'] == 0] = self.opt.ignore_index
        ret["meta"] = data_path
        return ret
