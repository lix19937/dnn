import shutil

from util import path_change, is_float
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from align_ori_data import get_pcd_path, get_pay_path, get_bin_path
from util import read_ascii_pcd

import json
from scipy.spatial.transform import Rotation as R


def parse_to_ascii():
    src = '/home/igs/use_dir/seg_data/source_data/label_20221109/label_bin'
    data_list = sorted(os.listdir(src))
    for data in tqdm(data_list):
        path = osp.join(src, data)
        data = np.fromfile(path, np.uint8)
        data[data == 5] = 9
        data.tofile(path)

def new_data():
    src = '/home/igs/mnt/lidar_label/算法_数据标注/Lidar_OD_SEG'
    tgt_root = '/data/seg_data/ori_data/'
    date_list = ['label_20220329']
    for date in date_list:
        print(date)
        tgt_path = path_change(osp.join(src, date, 'SEG'))
        if not osp.exists(tgt_path):
            print(date + ' not exist SEG dir!')
            continue
        if 'SEG1' in os.listdir(tgt_path):
            tgt_path = osp.join(tgt_path, 'SEG1')
        for root, dirs, files in os.walk(tgt_path):
            if len(files) > 3 and is_float(files[0][:-4]):
                break
        root = osp.dirname(root)
        for root, dirs, files in os.walk(root):
            break
        dirs.sort()
        for dir_path in dirs:
            dir = osp.join(root, dir_path)
            data_list = os.listdir(dir)
            data_list = sorted([_ for _ in data_list if '.pcd' in _])
            if len(data_list) == 0:
                continue
            # data_list = sorted(os.listdir(dirs))
            tgt_dir = osp.join(tgt_root, date)
            tgt_pcd_dir = osp.join(tgt_dir, 'lidar')
            tgt_label_dir = osp.join(tgt_dir, 'label_bin')
            tgt_calib_dir = osp.join(tgt_dir, 'calibration')
            tgt_jpg_dir = osp.join(tgt_dir, 'camera', 'Front120')
            os.makedirs(tgt_pcd_dir, exist_ok=True)
            os.makedirs(tgt_label_dir, exist_ok=True)
            os.makedirs(tgt_jpg_dir, exist_ok=True)
            os.makedirs(tgt_calib_dir, exist_ok=True)
            f_pcd = open(osp.join(tgt_dir, 'only_pcd.txt'), 'w')
            f_jpg = open(osp.join(tgt_dir, 'no_jpg.txt'), 'w')
            out_length = 0
            for data in tqdm(data_list):
                stamp = data[:-4]
                # ymd = time.strftime('%Y%m%d', time.localtime(float(stamp)))
                pcd_path = osp.join(dir, data)
                pay_path = get_bin_path(stamp)
                jpg_path = pcd_path.replace('.pcd', '.jpg')
                if pay_path is None:
                    f_pcd.write(pcd_path + '\n')
                    continue
                if not osp.exists(jpg_path):
                    f_jpg.write(pcd_path + '\n')
                    continue
                cali_path = osp.dirname(pcd_path) + '.yaml'
                if not osp.exists(cali_path):
                    idx = cali_path.find('/SEG/')
                    cali_path = cali_path[:idx] + '/Luminar_2_Vehicle.yaml'
                if cali_path is None:
                    print(f'{stamp} not find cali_path')
                    continue
                pcd = read_ascii_pcd(pcd_path)
                pcd.save_pcd(osp.join(tgt_pcd_dir, stamp + '.pcd'))
                # shutil.copy(pcd_path, tgt_pcd_dir)
                # shutil.copy(pay_path, tgt_label_dir)
                shutil.copy(jpg_path, tgt_jpg_dir)
                shutil.copy(cali_path, osp.join(tgt_calib_dir, stamp + '.yaml'))
                pay_path = pay_path.replace('pcd_dir', 'ann_dir')
                label = np.fromfile(pay_path, np.uint8)
                label[label > 18] = 18
                label.tofile(os.path.join(tgt_label_dir, f'{stamp}.bin'))
                out_length += 1
            f_jpg.close()
            f_pcd.close()
            print(f'ori size is {len(data_list)}, out size is {out_length}')

if __name__ == '__main__':
    parse_to_ascii()
