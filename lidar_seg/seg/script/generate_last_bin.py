import os
import os.path as osp
import shutil
import time
from tqdm import tqdm
import numpy as np
from pcd_io import *
from convert_luminar_seg import get_rotate

root_dir = '/data/luminar_seg/baffle_0527'
pcd_dir = osp.join(root_dir, 'pcd_dir')
last_pcd_dir = osp.join(root_dir, 'last_pcd_dir')
odom_data = np.loadtxt(osp.join(root_dir, 'OD_parse.txt'))
stamp_list = odom_data[:, 0]
velo_data = odom_data[:, [7,10]]
odom_length = stamp_list.shape[0]
cali_path = '/home/igs/mnt/lidar_label/算法_数据标注/Lidar_OD_SEG/数据送标/2022/202205/label_20220520/Luminar_2_Vehicle.yaml'
tf = get_rotate(cali_path)

data_list = sorted(os.listdir(pcd_dir))

def pcd_to_bin(pcd_path, bin_path, tf):
    pcd = point_cloud_from_path(pcd_path)
    pcd_np_points = np.ones((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data['x'])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data['y'])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data['z'])
    cloud_trans = np.matmul(tf, pcd_np_points.T).T[:, :3].astype(np.float32)
    pcd_np_points[:, :3] = cloud_trans
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data['intensity'])
    pcd_np_points.tofile(bin_path)

def generate_last_bin():
    tgt_dir = osp.join(root_dir, 'last_pcd_dir')
    last_odom_dir = osp.join(root_dir, 'last_odom_dir')
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(last_odom_dir, exist_ok=True)

    src_dir = '/home/igs/mnt/lidar_label/算法_数据标注/Lidar_OD_SEG/数据送标/2022/\
202205/label_20220520/20221018/OD/sample/sampling_for_od/'

    ori_list = []
    ori_path_list = []
    for task in os.listdir(src_dir):
        task_path = osp.join(src_dir, task)
        tmp_list = os.listdir(task_path)
        ori_list.extend([_[:-4] for _ in tmp_list if 'pcd' in _])
        ori_path_list.extend([osp.join(src_dir, task, _) for _ in tmp_list if 'pcd' in _])

    ori_path_dict = dict(zip(ori_list, ori_path_list))
    ori_list.sort()
    # ori_path_list.sort()
    
    j = 0
    # f = open(osp.join(root_dir, 'odometry.txt'), 'w')
    for data_path in tqdm(data_list):
        stamp = data_path.split('_')[0]
        idx = ori_list.index(stamp)
        if idx == 0:
            last = 0
        else:
            last = idx - 1
        last_path = ori_list[last]
        last_stamp = float(last_path[:20])
        if not 0 < (float(stamp) - last_stamp) < 0.25:
            print(f'{stamp} and {last_stamp} not in (0, 0.25) range!!')
            pcd_to_bin(ori_path_dict[last_path], osp.join(tgt_dir, data_path), tf)
            with open(osp.join(last_odom_dir, data_path), 'w') as f:
                f.write(f'{last_stamp} 0 0 0\n')
            continue
        while j < odom_length:
            if stamp_list[j] > last_stamp:
                break
            j += 1
        diff0 = abs(stamp_list[j - 1] - last_stamp)
        diff1 = abs(stamp_list[j] - last_stamp)
        idx = j if diff0 > diff1 else j - 1
        with open(osp.join(last_odom_dir, data_path), 'w') as f:
                f.write(f'{last_stamp} {velo_data[idx, 0]} {velo_data[idx, 1]} 1\n')
        pcd_to_bin(ori_path_dict[last_path], osp.join(tgt_dir, data_path), tf)
    f.close()

generate_last_bin()