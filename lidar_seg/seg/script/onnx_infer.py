from time import sleep
import onnx
import onnxruntime as rt
import os
import sys
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import open3d as o3d

color_dict = [[255, 0, 0], [255, 127, 80], [60, 179, 113],
           [200, 200, 200], [107, 142, 35], [152, 251, 152], [100, 60, 255],
           [0, 0, 255], [255, 255, 0]]

class SegData:
    def __init__(self) -> None:
        self.img_scale=(5, 192, 1024)
        fov_angle=(-16, 8)
        horizon_angle=(-64, 64)
        fov_up = fov_angle[1] / 180.0 * np.pi  # field of view up in rad
        self.fov_down = fov_angle[0] / 180.0 * np.pi  # field of view down in rad
        self.fov = fov_up - self.fov_down  # get field of view total in rad
        self.fov_left = horizon_angle[0] / 180.0 * np.pi
        self.horizon_fov = horizon_angle[1] / 180.0 * np.pi - self.fov_left

        quat_xyzw = [0, 0, 0, 1]
        self.translation = np.array([0, 0, 2.0], np.float32)
        self.rotate = R.from_quat(quat_xyzw).as_matrix()

    def _in_roi(self, index):
        return 0 <= index[0] < self.img_scale[1] and \
            0 <= index[1] < self.img_scale[2]

    def project_to_img(self, ori_cloud):
        cloud = (np.matmul(self.rotate.T, ori_cloud.T) -
            self.translation.reshape(3, -1)).T.astype(np.float32)
        cloud_size = cloud.shape[0]
        scan_x = cloud[:, 0]
        scan_y = cloud[:, 1]
        scan_z = cloud[:, 2]
        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        depth = np.linalg.norm(cloud, 2, axis=1)
        pitch = np.arcsin(scan_z / depth)
        c = ((yaw - self.fov_left) / self.horizon_fov *
            self.img_scale[2]).astype(np.int32)  # in [0.0, 1.0]
        r = (self.img_scale[1] - (pitch - self.fov_down) / self.fov *
            self.img_scale[1]).astype(np.int32)  # in [0.0, 1.0]

        # mask = np.ones((1, cloud_size), np.int32)
        dict_mat = dict()
        for idx in range(cloud_size):
            tmp_tuple = (r[idx], c[idx])
            if tmp_tuple not in dict_mat:
                dict_mat[tmp_tuple] = [idx]
            else:
                dict_mat[tmp_tuple].append(idx)
        img_ori = np.zeros(self.img_scale, 'f')
        for key, val in dict_mat.items():
            if self._in_roi(key):
                img_ori[0, key[0], key[1]] = ori_cloud[:, 0][val[-1]]
                img_ori[1, key[0], key[1]] = ori_cloud[:, 1][val[-1]]
                img_ori[2, key[0], key[1]] = ori_cloud[:, 2][val[-1]]
                img_ori[3, key[0], key[1]] = len(val)
                img_ori[4, key[0], key[1]] = depth[val[-1]]

        return np.expand_dims(img_ori, axis=0)

def infer_onnx(sess, dummy_input):
    mm_inputs = {"input": dummy_input}
    onnx_result = sess.run(None, mm_inputs)
    return onnx_result

def generate_label_gt(data_dir, onnx_path):
    bin_dir = osp.join(data_dir, 'bin_dir')
    img_dir = osp.join(data_dir, 'img_dir')
    label_dir = osp.join(data_dir, 'label_dir')
    prob_dir = osp.join(data_dir, 'prob_dir')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    bin_list = os.listdir(bin_dir)
    segData = SegData()
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    sess = rt.InferenceSession(onnx_path)
    for bin_file in tqdm(bin_list):
        bin_path = osp.join(bin_dir, bin_file)
        pc = np.fromfile(bin_path, np.float32).reshape(-1, 3)
        img = segData.project_to_img(pc)
        infer_result = infer_onnx(sess, img)
        infer_result[0].tofile(osp.join(label_dir, bin_file))
        infer_result[1].tofile(osp.join(prob_dir, bin_file))
        img.tofile(osp.join(img_dir, bin_file))

def show_pc_label(data_dir):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pointcloud = o3d.geometry.PointCloud()
    to_reset = True
    vis.add_geometry(pointcloud)
    img_dir = osp.join(data_dir, 'img_dir')
    label_dir = osp.join(data_dir, 'label_dir')
    img_list = os.listdir(img_dir)
    for img_file in tqdm(img_list):
        img_path = osp.join(img_dir, img_file)
        label_path = osp.join(label_dir, img_file)
        data = np.fromfile(img_path, np.float32).reshape(5, -1)
        label = np.fromfile(label_path, np.int64)
        mask = data[3, :] > 0.1
        pc = data[:3, mask].T
        pc_label = label[mask]
        rgb = np.ones((pc.shape[0], 3), np.float32)
        for t in range(pc.shape[0]):
            rgb[t, :] = np.asarray(color_dict[pc_label[t]], np.float32) / 255
        pointcloud.points = o3d.utility.Vector3dVector(pc)
        pointcloud.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pointcloud)
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
            vis.run()
        else:
            sleep(0.5)
            vis.poll_events()
            vis.update_renderer()

def infer_gt(data_dir, onnx_path):
    img_dir = osp.join(data_dir, 'img_dir')
    label_dir = osp.join(data_dir, 'label_dir')
    prob_dir = osp.join(data_dir, 'prob_dir')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    # os.makedirs(img_dir, exist_ok=True)
    img_list = os.listdir(img_dir)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    sess = rt.InferenceSession(onnx_path)
    for img_file in tqdm(img_list):
        img_path = osp.join(img_dir, img_file)
        img = np.fromfile(img_path, np.float32).reshape(-1, 5, 192, 1024)
        infer_result = infer_onnx(sess, img)
        infer_result[0].tofile(osp.join(label_dir, img_file))
        infer_result[1].tofile(osp.join(prob_dir, img_file))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"usage: python {__file__} DATA_DIR")
        exit(-1)
    root_dir = sys.argv[1]
    # infer_gt(root_dir, osp.join(root_dir, 'lidarnet_seg.onnx'))
    show_pc_label(root_dir)