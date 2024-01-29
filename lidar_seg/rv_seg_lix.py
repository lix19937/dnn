from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import os.path as osp
from scipy.spatial.transform import Rotation as R

lidar_seg_gt_map = {0:0,  1:1,  2:1,  3:2,  4:2,  5:2,  6:3,   7:4,  8:2, 9:5, 10:6,
                    11:2, 12:7, 13:8, 14:2, 15:3, 16:3, 17:10, 18:9, 19:3}

def label_change(label):
    label[label > 19] = 17
    label = np.vectorize(lidar_seg_gt_map.get)(label).astype(np.uint8)
    return label

class RvSeg(data.Dataset):
    CLASSES = ('路面', '人行道', '障碍物', '高空噪声', '支撑杆', '交通牌', '交通锥',
               '车或者人', '轮挡', '挡板', '红绿灯', '未知目标',)

    PALETTE = [[255, 0, 0], [255, 127, 80], [0, 255, 127],
        [200, 200, 200], [107, 142, 35], [0, 255, 127],
        [152, 251, 152], [0, 0, 255], [142, 0, 252], [119, 11, 32], [128, 64, 128]]

    def __init__(self) -> None:
        super().__init__()
        self.img_scale=(5, 192, 1024) # c h w
        fov_angle=(-16, 8)
        horizon_angle=(-64, 64)
        fov_up = fov_angle[1] / 180.0 * np.pi         # field of view up in rad
        self.fov_down = fov_angle[0] / 180.0 * np.pi  # field of view down in rad
        self.fov = fov_up - self.fov_down             # get field of view total in rad
        self.fov_left = horizon_angle[0] / 180.0 * np.pi
        self.horizon_fov = horizon_angle[1] / 180.0 * np.pi - self.fov_left

        # quat_xyzw = [-0.0061, -0.01, 0.002759, 1]
        # self.translation = np.array([1.633116386017969, 0.023100567710573547, 1.5548923643842074], np.float32)

        quat_xyzw = [0, 0, 0, 1]
        self.translation = np.array([0, 0, 2.0], np.float32)
        self.rotate = R.from_quat(quat_xyzw).as_matrix()
         
    def _in_roi(self, index):
        return 0 <= index[0] < self.img_scale[1] and 0 <= index[1] < self.img_scale[2]

    def _augment(self, lidar):
        euler = np.random.randint(-10, 10, size=3).astype(np.float) / 100
        euler[0] = 0
        from scipy.spatial.transform.rotation import Rotation as R
        rot = R.from_euler('zyx', euler, degrees=True).as_matrix().astype(np.float32)
        transformed_cloud = np.matmul(rot, lidar[:, :3].transpose()).astype(np.float32)
        return transformed_cloud.T

     # [N, 3]  [1, NUM_CLASS]
     # ori_cloud ziche zuo biao xi 
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

        # prj to img coor index of col  row  
        c = ((yaw - self.fov_left) / self.horizon_fov * self.img_scale[2]).astype(np.int32)  # [N, ]
        r = (self.img_scale[1] - (pitch - self.fov_down) / self.fov * self.img_scale[1]).astype(np.int32)   # [N, ]

        dict_mat = dict()
        for idx in range(cloud_size):
            tmp_tuple = (r[idx], c[idx])
            if tmp_tuple not in dict_mat:
                dict_mat[tmp_tuple] = [idx] # just get intensity 
            else:
                dict_mat[tmp_tuple].append(idx)

        img_ori = np.zeros(self.img_scale, 'f')
        label_img = np.ones(self.img_scale[1:], np.uint8) * 10
        render_img = np.ones((*self.img_scale[1:], 3), np.uint8) * 0

        print("valid num:", len(dict_mat))
        for key, val in dict_mat.items():
            if self._in_roi(key): #  x, y  
                img_ori[0, key[0], key[1]] = ori_cloud[:, 0][val[-1]]
                img_ori[1, key[0], key[1]] = ori_cloud[:, 1][val[-1]]
                img_ori[2, key[0], key[1]] = ori_cloud[:, 2][val[-1]]
                img_ori[3, key[0], key[1]] = len(val)      #
                img_ori[4, key[0], key[1]] = depth[val[-1]]#
                label_img[key[0], key[1]] = label[0, val[-1]]

                render_img[key[0], key[1], ::-1] = self.PALETTE[label[0, val[-1]]]
                
        results = dict()
        results['input'] = img_ori
        results['gt_segment_label'] = label_img
        results['render_img'] = render_img
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

if __name__ == '__main__':
    rv = RvSeg()
    seg_label_path = "./1648609228.109629977_ML021_MW001_MT001.bin"
    gt_seg_label = np.fromfile(seg_label_path, np.uint8).reshape(1, -1)
    gt_seg_label = label_change(gt_seg_label)

    # Reads lidar data.
    data_path = "./1648609228.109629977_ML021_MW001_MT001.bin"
    lidar_one_frame = np.fromfile(data_path, np.float32).reshape(-1, 3)
    print(lidar_one_frame.shape)

    ret = rv.project_to_rv(lidar_one_frame, gt_seg_label)#####################
    pcd2img = ret['input']

    render_img = ret['render_img']
    print(pcd2img.shape, render_img.shape)
    import cv2
    cv2.imwrite("render_img.png", render_img)

    gt_seg_label = gt_seg_label.reshape(-1,1).squeeze() 

    print(gt_seg_label.shape)

    from lidar_eval_lix import show_worst_instance

    show_worst_instance(lidar_one_frame, gt_seg_label, data_path, rv.PALETTE)

    # data = np.ones((100, 100), np.uint8) * 255  
    # cv2.imwrite('output.png', data)
