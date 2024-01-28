import os
import open3d as o3d
import numpy as np
import cv2
from tqdm import tqdm
import time

class BevVis(object):
    def __init__(self, ranges=(0, 200, -200, 200, -3, 3), voxel_size=(0.3, 0.3, 0.2)):
        self.ranges = ranges
        self.voxel_size = voxel_size

    def generage_bev(self):
        bev_h = (self.ranges[1] - self.ranges[0]) // self.voxel_size[0] + 1
        bev_w = (self.ranges[3] - self.ranges[2]) // self.voxel_size[1] + 1
        bev = np.ones((int(bev_h), int(bev_w), 3)).astype(np.uint8)
        return bev

    def plot_pts(self, lidar, label, bev, color):
        inds = (lidar[:, 0] > self.ranges[0]) & (lidar[:, 0] < self.ranges[1]) & \
               (lidar[:, 1] > self.ranges[2]) & (lidar[:, 1] < self.ranges[3]) & \
               (lidar[:, 2] > self.ranges[4]) & (lidar[:, 2] < self.ranges[5])

        lidar = lidar[inds]
        label = label[inds]
        bev_h = (self.ranges[1] - self.ranges[0]) // self.voxel_size[0] + 1
        bev_w = (self.ranges[3] - self.ranges[2]) // self.voxel_size[1] + 1
        v = bev_h - (lidar[:, 0] - self.ranges[0]) / self.voxel_size[0]
        u = bev_w - (lidar[:, 1] - self.ranges[2]) / self.voxel_size[1]
        v = np.clip(v, 0, bev_h - 1)
        u = np.clip(u, 0, bev_w - 1)
        v = v.reshape(-1).astype(np.int)
        u = u.reshape(-1).astype(np.int)
        order = np.argsort(lidar[:, 2])
        bev[v[order], u[order], :] = [color[i] for i in label[order]]
        return bev

def bev(lidar, label, color_dict, img_dir, file_name, ori):
    img_path = os.path.join(img_dir, f'{file_name}.jpg')
    lidar_vis = BevVis()
    bev = lidar_vis.generage_bev()
    bev = lidar_vis.plot_pts(lidar, label, bev, color=color_dict)
    img = cv2.resize(bev, (2000, 1000), interpolation=cv2.INTER_CUBIC)
    if ori is not None:
        ori = cv2.resize(ori, (500, 250), interpolation=cv2.INTER_CUBIC)
        img[:250, :500, :] = ori
    cv2.imwrite(img_path, img)


def search(root_dir, file_list):
    for file in os.listdir(root_dir):
        if file.split('.')[-1] == 'bin':
            file_list.append(file)


def view_seg_label(file_list):
    # file_list = []
    # root_dir = out_dir
    label_dir = os.path.join(root_dir, 'reformat', 'ann_dir')
    pcd_dir = os.path.join(root_dir, 'reformat', 'pcd_dir')
    # search(label_dir, file_list)
    # file_list.sort()
    # palette_dir = os.path.join(root_dir, 'palette_luminar')
    # if not os.path.exists(palette_dir):
    #     os.makedirs(palette_dir)
    #     for i, name in enumerate(label_name):
    #         color = color_dict[i]
    #         img = np.zeros((100, 100, 3), np.uint8)
    #         img[:, :, 0] = color[0]
    #         img[:, :, 1] = color[1]
    #         img[:, :, 2] = color[2]
    #         cv2.imwrite(os.path.join(palette_dir, f'{i}_{name}.png'), img)
    img_dir = os.path.join(root_dir, 'seg_' + time.strftime("%y%m%d", time.localtime()))
    os.makedirs(img_dir, exist_ok=True)
    for file in tqdm(file_list):
        file_name = file[:17]
        pcd_path = os.path.join(pcd_dir, file)
        points = np.fromfile(pcd_path, np.float32).reshape(-1, 3)
        label_path = os.path.join(label_dir, file)
        label = np.fromfile(label_path, dtype=np.int8)
        img_path = os.path.join(tmp_img_dir, file_name + '.jpg')
        img = cv2.imread(img_path)
        bev(points, label, color_dict, img_dir, file_name, img)
    print('convert images done!')

def convert_9_to_6(file_name):
    return format(float(file_name[:-4]), '.6f')
    # return file_name[:-4]

label_name2 = ('road', 'sidewalk', 'terrain', 'lane', 'fence', 'wall',
           'building', 'suspension', 'pole', 'vegetation', 'traffic sign', 'traffic cone', 'object_other',
           'Traffic participants', 'wheel stop', 'barrier gate', 'reflection',
           'mist', 'others',)

label_name1 = ('road', 'sidewalk', 'terrain', 'fence', 'wall',
           'building', 'suspension', 'pole', 'vegetation', 'traffic sign', 'traffic cone', 'object_other',
           'vehicle', 'non-motor vehicle', 'person', 'wheel stop', 'barrier gate', 'reflection',
           'mist', 'others',)

label_name = ('road', 'sidewalk', 'terrain', 'fence', 'wall',
           'building', 'suspension', 'pole', 'vegetation', 'traffic_sign', 'traffic_cone', 'object_other',
           'Traffic_participants', 'wheel_stop', 'barrier_gate', 'reflection',
           'mist', 'others',)

color_dict = [[0, 165, 255], [255, 127, 80], [107, 142, 35],
           [255, 255, 0], [0, 200, 200], [128, 128, 0], [200, 200, 200],
           [107, 142, 35], [0, 255, 127], [152, 251, 152], [255, 60, 100], [220, 20, 60],
           [0, 0, 255], [60, 179, 113], [255, 215, 0], [0, 60, 100],
           [0, 80, 100], [72, 51, 159], [119, 11, 32], [128, 64, 128]]

# luminar_2_vehicle = np.array([[9.99924839e-01, -1.13726659e-02, -4.58624261e-03, 1.66035831e+00],
#     [1.14255212e-02, 9.99866664e-01, 1.16680861e-02, 3.09985187e-02],
#     [4.45293356e-03, -1.17196087e-02, 9.99921381e-01, 1.61256278e+00],
#      [0., 0., 0., 1.]])

def run():
    pc_list = []
    label_dict = {}
    for root, _, files in os.walk(label_dir):
        for file in files:
            if '.bin' in file:
                label_dict[file[:-4]] = root

    for root, _, files in os.walk(data_dir):
        for file in files:
            if '.pcd' in file and convert_9_to_6(file) in label_dict:
                    pc_list.append(os.path.join(root, file))
    if len(pc_list) == 0:
        t = list(label_dict.keys())
        t.sort()
        files.sort()
        print(f'no data matched, label_range is {t[0]} -> {t[-1]}, pcd_range is {files[0][:-4]} -> {files[-1][:-4]}')
        return
    local_out_dir = os.path.join(out_dir, 'reformat')
    img_dir = os.path.join(local_out_dir, "pcd_dir")
    ann_dir = os.path.join(local_out_dir, "ann_dir")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    pc_list.sort()
    out_list = []
    for pc_path in tqdm(pc_list):
        pcd = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd.points)
        points = np.c_[points, np.ones(points.shape[0])].astype(np.float32)
        cloud_trans = np.matmul(luminar_2_vehicle, points.T).T[:, :3].astype(np.float32)
        key = convert_9_to_6(os.path.basename(pc_path))
        label_path = os.path.join(label_dict[key], key + '.bin')
        scenes_path = os.path.join(label_dict[key], key + '.txt')
        label = np.fromfile(label_path, np.uint8)
        label[label == 3] = 0
        label[label > 3] = label[label > 3] - 1
        label[label >= 18] = 17
        if (points.shape[0] != label.shape[0]):
            print(f"points size {points.shape[0]} doesn't match label size {label.shape[0]}")
            continue
        scene = open(scenes_path, 'r').readline()
        scene = scene.replace(' ', '_')
        cloud_trans.tofile(os.path.join(img_dir, f'{key}_{scene}.bin'))
        label.tofile(os.path.join(ann_dir, f'{key}_{scene}.bin'))
        out_list.append(f'{key}_{scene}.bin')
    print('reformat done!')
    return out_list


def cp_filtered_data(task = 'reformat'):
    pc_dir = os.path.join(out_dir, task, 'pcd_dir')
    label_dir = os.path.join(out_dir, task, 'ann_dir')
    filter_path = os.path.join(out_dir, task, 'filter.txt')
    filter_id = []
    def str2i(line):
        return int(float(line) * 10)
    with open(filter_path, 'r') as f:
        text = f.readlines()
        text = [_.split(' ')[0] for _ in text]
        text = [_.split('→') for _ in text][1:]
        for line in text:
            try:
                start = str2i(line[0])
            except:
                continue
            if len(line) < 2:
                filter_id.append(start)
                continue
            end = str2i(line[1])
            for i in range(start, end + 1):
                filter_id.append(i)
            # for i in range(int(line[0]), int(line[-1]) + 1):
            #     filter_id.append(i)
    paths = os.listdir(pc_dir)
    paths.sort()
    # import shutil
    for path in tqdm(paths):
        id = str2i(path[:17])
        if id not in filter_id:
            continue
        # full_path = os.path.join(pc_dir, path)
        # cloud = np.fromfile(full_path, np.float32).reshape(-1, 3)
        # cloud[:, 3] = 1
        # cloud_trans = np.matmul(np.array(luminar_2_vehicle), cloud.T).T[:, :3].astype(np.float32)
        # new_path = str(id + start).rjust(6, '0') + path[6:]
        # cloud_trans.tofile(os.path.join(out_dir, 'reformat', 'pcd_dir', new_path))
        os.remove(os.path.join(pc_dir, path))
        os.remove(os.path.join(label_dir, path))
        # shutil.move(os.path.join(pc_dir, path), os.path.join(out_dir, 'reformat', 'pcd_dir', path))
        # shutil.move(os.path.join(label_dir, path), os.path.join(out_dir, 'reformat', 'ann_dir', path))
    os.remove(filter_path)
        

if __name__ == "__main__":
    root_dir = "/data/seg_pay/0304"
    out_dir = root_dir
    tasks = ['pcd_seg1','pcd_seg3', 'pcd_seg4', 'pcd_seg2']
    label_tasks = ['seg1','seg3','seg4', 'seg2']
    from scipy.spatial.transform import Rotation as R
    with open(os.path.join(root_dir, 'Luminar_2_Vehicle.yaml'), 'r') as f:
        lines = f.read().splitlines()
    lines = [_.split(',') for _ in lines]
    lines = [[_.strip() for _ in line] for line in lines]
    quat = [float(lines[6][1]), float(lines[7][0]), float(lines[7][1][:-2]), float(lines[6][0][8:])]
    translation = [float(lines[12][0][8:]), float(lines[12][1]), float(lines[13][0][:-2])]
    luminar_2_vehicle = np.eye(4)
    luminar_2_vehicle[:3, :3] = R.from_quat(quat).as_matrix()
    luminar_2_vehicle[:3, 3] = translation
    img_dir = '/home/igs/mnt/lidar_label/算法_数据标注/Lidar OD&SEG/label_20220304/SEG'
    for i in range(len(tasks)):
        data_dir = os.path.join(root_dir, tasks[i])
        label_dir = os.path.join(root_dir, label_tasks[i])
        pc_list = run()
        tmp_img_dir = os.path.join(img_dir, label_tasks[i])
        view_seg_label(pc_list)
    # cp_filtered_data('reformat')
