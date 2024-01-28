
# -*- coding:utf-8 -*-
import shutil
import os
import os.path as osp
import time
from util import is_float, path_change
from tqdm import tqdm

tgt_root = '/data/seg_data/table'

def get_ori_path(text):
    date = time.strftime('%Y%m%d', time.localtime(float(text)))
    tgt_dir = osp.join(tgt_root, date)
    train_pcd = osp.join(tgt_dir, 'lidar.txt')
    if not osp.exists(train_pcd):
        return None
    with open(train_pcd, 'r') as f:
        tgt_list = f.readlines()
        tgt_dict = dict(zip([_.split('/')[-1].strip()[:-4] for _ in tgt_list], tgt_list))
        if str(text) not in tgt_dict:
            return None
        path = tgt_dict[str(text)]
        # print(path)
    return path.strip()

def get_pay_path(text):
    date = time.strftime('%Y%m%d', time.localtime(float(text)))
    tgt_dir = osp.join(tgt_root, date)
    train_pcd = osp.join(tgt_dir, 'train_pay.txt')
    if not osp.exists(train_pcd):
        return None
    with open(train_pcd, 'r') as f:
        tgt_list = f.readlines()
        tgt_dict = dict(zip([_.split('/')[-1].strip() for _ in tgt_list], tgt_list))
        if str(text) not in tgt_dict:
            return None
        path = tgt_dict[str(text)]
        # print(path)
    return path.strip()

def get_bin_path(text):
    tgt_root = '/data/seg_align'
    date = time.strftime('%Y%m%d', time.localtime(float(text)))
    tgt_dir = osp.join(tgt_root, date)
    train_bin = osp.join(tgt_dir, 'train_bin.txt')
    if not osp.exists(train_bin):
        return None
    with open(train_bin, 'r') as f:
        tgt_list = f.readlines()
        tgt_dict = dict(zip([_.split('/')[-1].strip().split('_')[0] for _ in tgt_list], tgt_list))
        if str(text) not in tgt_dict:
            return None
        path = tgt_dict[str(text)]
        # print(path)
    return path.strip()

def get_pcd_path(text):
    tgt_root = '/data/seg_data/table'
    date = time.strftime('%Y%m%d', time.localtime(float(text)))
    tgt_dir = osp.join(tgt_root, date)
    train_bin = osp.join(tgt_dir, 'lidar.txt')
    if not osp.exists(train_bin):
        return None
    with open(train_bin, 'r') as f:
        tgt_list = f.readlines()
        tgt_dict = dict(zip([_.split('/')[-1].strip()[:-4] for _ in tgt_list], tgt_list))
        if str(text) not in tgt_dict:
            return None
        path = tgt_dict[str(text)]
        # print(path)
    return path.strip()

def give_jpg(data_dir):
    pcd_list = os.listdir(osp.join(data_dir, 'pcd_dir'))
    jpg_dir = osp.join(data_dir, 'jpg_dir')
    os.makedirs(jpg_dir, exist_ok=True)
    for pcd in tqdm(pcd_list):
        timestamp = pcd.split('_')[0]
        if timestamp[0] == '0':
            continue
        p = get_ori_path(timestamp)
        if p is None:
            print(f"don't find {pcd} ori path!")
            continue
        p = p.replace('pcd', 'jpg')
        if not osp.exists(p):
            print(f"{p} not found!")
            continue
        shutil.copy(p, jpg_dir)

def align_pcd():
    src = '/home/igs/mnt/lidar_label/算法_数据标注/Lidar_OD_SEG'
    # tgt_list = os.listdir(src)
    # tgt_list = sorted([_ for _ in tgt_list if 'label_2022' in _])
    tgt_list = ['label_20220919', 'label_20220926', 'label_20221009', 'label_20221017',
    'label_20221024', 'label_20221031', 'label_20221109', 'label_20221117']

    for tgt in tgt_list:
        print(tgt)
        tgt_path = path_change(osp.join(src, tgt, 'SEG'))
        if not osp.exists(tgt_path):
            print(tgt + ' not exist SEG dir!')
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
            file_list = os.listdir(dir)
            file_list = sorted([_ for _ in file_list if '.pcd' in _])
            if len(file_list) == 0:
                continue
            last_date = ''
            f = None
            for file in file_list:
                date = time.strftime('%Y%m%d', time.localtime(float(file[:-4])))
                if date != last_date:
                    tgt_dir = osp.join(tgt_root, date)
                    os.makedirs(tgt_dir, exist_ok=True)
                    last_date = date
                    if f is not None:
                        f.close()
                    f = open(osp.join(tgt_dir, 'train_pcd.txt'), 'a')
                f.write(osp.join(dir, file) + '\n')
            if f is not None:
                f.close()

def align_pay():
    tgt_root = '/home/igs/use_dir/data_pay/all_unzip/'
    src_root = '/data/seg_align'
    for i in [4]:
        date = str(i) + '月交付'
        p1 = osp.join(tgt_root, date)
        tasks = os.listdir(p1)
        for task in tasks:
            p2 = osp.join(p1, task, '单帧')
            if not osp.exists(p2):
                continue
            # p2 = osp.dirname(p2)
            labels = sorted(os.listdir(p2))
            for label in labels[3:]:
                p3 = osp.join(p2, label)
                for root, dirs, _ in os.walk(p3):
                    if not (len(dirs) > 0 and is_float(dirs[0]) and float(dirs[0]) > 1000000000):
                        continue
                    dirs.sort()
                    if len(dirs) == 0:
                        continue
                    last_date = ''
                    f = None
                    print(root)
                    for file in tqdm(dirs):
                        date = time.strftime('%Y%m%d', time.localtime(float(file[:-4])))
                        if date != last_date:
                            tgt_dir = osp.join(src_root, date)
                            os.makedirs(tgt_dir, exist_ok=True)
                            last_date = date
                            if f is not None:
                                f.close()
                            f = open(osp.join(tgt_dir, 'train_pay.txt'), 'a')
                        f.write(osp.join(root, file) + '\n')
                    if f is not None:
                        f.close()

def align_bin():
    root = '/data/seg_data/ori_data/'
    task_list = sorted(os.listdir(root))
    for task in task_list:
        pcd_dir = osp.join(root, task, 'lidar')
        print(f'process {pcd_dir} now...')
        bin_list = sorted(os.listdir(pcd_dir))
        last_date = ''
        f = None
        for bin in tqdm(bin_list):
            date = time.strftime('%Y%m%d', time.localtime(float(bin[:-4])))
            if date != last_date:
                tgt_dir = osp.join(tgt_root, date)
                os.makedirs(tgt_dir, exist_ok=True)
                last_date = date
                if f is not None:
                    f.close()
                f = open(osp.join(tgt_dir, 'lidar.txt'), 'a')
            f.write(osp.join(pcd_dir, bin) + '\n')
        if f is not None:
            f.close()

def align_filter():
    root = '/data/luminar_seg/'
    tgt_root = '/data/seg_data/ori_data'
    task_list = sorted(os.listdir(root))
    for task in task_list[9:]:
        filter_path = f'/data/luminar_seg/{task}/filter.txt'
        if not osp.exists(filter_path):
            print(filter_path + ' not exists!')
            continue
        print(f'process {filter_path} now...')
        bin_list = open(filter_path, 'r').readlines()
        bin_list = sorted([_.strip().split('_')[0] for _ in bin_list])
        last_date = ''
        f = None
        for bin in tqdm(bin_list):
            pcd_path = get_ori_path(bin)
            if pcd_path is None:
                continue
            date = pcd_path.split('/')[-3]
            # date = time.strftime('%Y%m%d', time.localtime(float(bin.split('_')[0])))
            if date != last_date:
                tgt_dir = osp.join(tgt_root, date)
                # os.makedirs(tgt_dir, exist_ok=True)
                last_date = date
                if f is not None:
                    f.close()
                f = open(osp.join(tgt_dir, 'filter.txt'), 'a')
            bin_path = bin + '.pcd'
            f.write(f'{bin_path}\n')
        if f is not None:
            f.close()

if __name__ == '__main__':
    # task_list = ['luminar_seg_0417', 'luminar_seg_0422', 'luminar_seg',
    # 'luminar_seg_0426', 'luminar_seg_0429', 'luminar_seg_0415', 'baffle_0527']

    # align_bin(task_list)
    align_pay()
    # give_jpg('/data/luminar_seg/luminar_seg_0415')