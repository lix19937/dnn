import os
import os.path as osp
from tqdm import tqdm
import shutil
from align_ori_data import get_bin_path
from util import read_ascii_pcd
import numpy as np

def main():
    img_dir = '/home/igs/use_dir/data_send/2022/202210/label_20221017/SEG/SEG1'
    out_dir = '/home/igs/use_dir/source_data/label_20221017'
    date_list = ['PP60TEST_202209120750-雨天城区']
    lidar_dir = osp.join(out_dir, 'lidar')
    jpg_dir = osp.join(out_dir, 'camera', 'Front120')
    cali_dir = osp.join(out_dir, 'calibration')
    os.makedirs(lidar_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(cali_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, 'label_bin'), exist_ok=True)
    for date in date_list:
        root_dir = osp.join(img_dir, date)
        cali_path = osp.join(img_dir, date + '.yaml')
        label_list = os.listdir(root_dir)
        pcd_list = [_ for _ in label_list if '.pcd' in _]
        pcd_list.sort()
        for pcd_file in tqdm(pcd_list):
            pcd_path = os.path.join(root_dir, pcd_file)
            pcd = read_ascii_pcd(pcd_path, True)
            pcd.save_pcd(osp.join(lidar_dir, pcd_file))
            shutil.copy(osp.join(root_dir, pcd_file.replace('.pcd', '.jpg')), jpg_dir)
            shutil.copy(cali_path, osp.join(cali_dir, pcd_file.replace('.pcd', '.yaml')))
        print(f'\033[0;32;40m{date} reformat done!\033[0m')

lidar_seg_gt_map = {0:0, 1:1, 2:1, 3:3, 4:12, 5:12, 6:12, 7:7, 8:8, 9:12, 10:10,
          11:11, 12:12, 13:13, 14:14, 15:15, 16:7, 17:7, 18:7, 19:19, 20:20}

def merge_label():
    root = '/home/igs/use_dir/source_data/label_20221109/label_bin'
    data_list = sorted(os.listdir(root))
    for data in tqdm(data_list):
        src_path = osp.join(root, data)
        label = np.fromfile(src_path, np.uint8)
        label = np.vectorize(lidar_seg_gt_map.get)(label)
        label.tofile(src_path)

if __name__ == '__main__':
    merge_label()
