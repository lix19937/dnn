import numpy as np
from scipy.spatial.transform import Rotation as R

class RvSeg:
    def __init__(self, ignore) -> None:
        super().__init__()
        self.img_scale=(5, 192, 1024)
        fov_angle=(-16, 8)
        horizon_angle=(-64, 64)
        fov_up = fov_angle[1] / 180.0 * np.pi  # field of view up in rad
        self.fov_down = fov_angle[0] / 180.0 * np.pi  # field of view down in rad
        self.fov = fov_up - self.fov_down  # get field of view total in rad
        self.fov_left = horizon_angle[0] / 180.0 * np.pi
        self.horizon_fov = horizon_angle[1] / 180.0 * np.pi - self.fov_left
        self.ignore_index = ignore

        quat_xyzw = [0, 0, 0, 1]
        self.translation = np.array([0, 0, 2.0], np.float32)
        self.rotate = R.from_quat(quat_xyzw).as_matrix()

    def _in_roi(self, index):
        return 0 <= index[0] < self.img_scale[1] and 0 <= index[1] < self.img_scale[2]

    def project_to_rv(self, cloud, label):
        """Resize images with ``results['scale']``."""
        # ori_cloud = cloud.copy()
        # ori_cloud[:, -1] = 1
        ori_cloud = cloud[:, :3]
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
        
        dict_mat = dict()
        for idx in range(cloud_size):
            tmp_tuple = (r[idx], c[idx])
            if tmp_tuple not in dict_mat:
                dict_mat[tmp_tuple] = [idx]
            else:
                dict_mat[tmp_tuple].append(idx)

        img_ori = np.zeros(self.img_scale, 'f')
        label_img = np.ones(self.img_scale[1:], np.uint8) * self.ignore_index
        for key, val in dict_mat.items():
            if self._in_roi(key):
                img_ori[0, key[0], key[1]] = ori_cloud[:, 0][val[-1]]
                img_ori[1, key[0], key[1]] = ori_cloud[:, 1][val[-1]]
                img_ori[2, key[0], key[1]] = ori_cloud[:, 2][val[-1]]
                img_ori[3, key[0], key[1]] = len(val)
                img_ori[4, key[0], key[1]] = depth[val[-1]]
                label_img[key[0], key[1]] = label[0, val[-1]]
                
        return np.expand_dims(img_ori, axis=0), label_img

    def project_to_img(self, data_path):
        lidar_one_frame = np.fromfile(data_path, np.float32).reshape(-1, 3)
        ann_path = data_path.replace('pcd_dir', 'ann_dir')
        gt_seg_label = np.fromfile(ann_path, np.uint8).reshape(1, -1)
        lidar_one_frame = np.hstack((lidar_one_frame, np.zeros((lidar_one_frame.shape[0], 1), np.float32)))

        return self.project_to_rv(lidar_one_frame, gt_seg_label)
