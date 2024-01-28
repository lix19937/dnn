import os
import os.path as osp
import numpy as np
from util import is_float, read_ascii_pcd
from align_ori_data import path_change
from tqdm import tqdm
import shutil

def label_change(label):
    # label[label == 3] = 0
    # label[label > 3] = label[label > 3] - 1
    label[label > 20] = 18
    return label


def get_rotate(path):
    from scipy.spatial.transform import Rotation as R
        
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    lines = [_.split(',') for _ in lines]
    lines = [[_.strip() for _ in line] for line in lines]
    quat = [float(lines[6][1]), float(lines[7][0]), float(lines[7][1][:-2]), float(lines[6][0][8:])]
    translation = [float(lines[12][0][8:]), float(lines[12][1]), float(lines[13][0][:-2])]
    luminar_2_vehicle = np.eye(4)
    luminar_2_vehicle[:3, :3] = R.from_quat(quat).as_matrix()
    luminar_2_vehicle[:3, 3] = translation
    return luminar_2_vehicle

def run():
    label_dict = {}
    for root, dirs, files in os.walk(label_dir):
        if len(dirs) < 5 or not is_float(dirs[0]):
            continue
        break
    root = osp.dirname(root)
    task_list = os.listdir(root)
    for task in task_list:
        dir_path = osp.join(root, task)
        dirs = os.listdir(dir_path)
        for dir in dirs:
            label_dict[dir] = os.path.join(dir_path, dir)
    pay_length = len(label_dict)
    for root, dirs, files in os.walk(data_dir):
        if len(files) < 5 or not is_float(files[0]):
            continue
        break
    root = osp.dirname(root)
    for root, dirs, files in os.walk(root):
        break
    img_dir = os.path.join(out_dir, "lidar")
    ann_dir = os.path.join(out_dir, "label_bin")
    jpg_dir = os.path.join(out_dir, "camera", 'Front120')
    scene_dir = os.path.join(out_dir, "scene")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(scene_dir, exist_ok=True)
    out_length = 0
    for task in dirs:
        file_path = osp.join(root, task)
        print(file_path)
        cali_path = osp.join(root, task + '.yaml')
        luminar_2_vehicle = get_rotate(cali_path)
        files = os.listdir(file_path)
        files = [_ for _ in files if '.pcd' in _ and _[:-4] in label_dict]
        if len(files) == 0:
            continue
        for pc_path in tqdm(files):
            pc_path = os.path.join(file_path, pc_path)
            pcd = read_ascii_pcd(pc_path)
            points = pcd.pc_data
            n_pc = np.c_[points['x'], points['y'], points['z'], np.ones(points.shape[0])].astype(np.float32)
            cloud_trans = np.matmul(luminar_2_vehicle, n_pc.T).T[:, :3].astype(np.float32)
            points['x'] = cloud_trans[:, 0]
            points['y'] = cloud_trans[:, 1]
            points['z'] = cloud_trans[:, 2]
            key = os.path.basename(pc_path)[:-4]
            label_path = os.path.join(label_dict[key], key + '.bin')
            scenes_path = os.path.join(label_dict[key], key + '.txt')
            label = np.fromfile(label_path, np.uint8)
            label[label > 20] = 18
            if (n_pc.shape[0] != label.shape[0]):
                print(f"points size {n_pc.shape[0]} doesn't match label size {label.shape[0]}")
                continue
            # scene = open(scenes_path, 'r').readline()
            # scene = scene.replace(' ', '_')
            # pcd.data = 'ascii'
            pcd.save_pcd(os.path.join(img_dir, f'{key}.pcd'))
            # cloud_trans.tofile(os.path.join(img_dir, f'{key}_{scene}.bin'))
            shutil.copy(scenes_path, scene_dir)
            shutil.copy(pc_path.replace('.pcd', '.jpg'), os.path.join(jpg_dir, key + '.jpg'))
            label.tofile(os.path.join(ann_dir, f'{key}.bin'))
            out_length += 1
    print(f'pay size is {pay_length}, out size is {out_length}')

if __name__ == "__main__":
    date_dir = "/home/igs/use_dir/data_pay/all_unzip/11月交付/11.27-12.1"
    img_dir = '/home/igs/mnt/lidar_label/算法_数据标注/Lidar_OD_SEG/'
    out_dir = '/home/igs/use_dir/source_data/traffic_light'
    date_list = ['Lidar_SEG-11.26-8369帧']
    
    for date in date_list:
        root_dir = osp.join(date_dir, date)
        label_list = os.listdir(root_dir)
        label_list.sort()
        for path in label_list[1:]:
            img_task_path = os.path.join(img_dir, path)
            label_dir = os.path.join(root_dir, path)
            # img_task_list = os.listdir(img_task_path)
            data_dir = path_change(os.path.join(img_task_path, 'SEG', 'SEG1'))
            run()
            print(f'{path} reformat done!')
        print(f'\033[0;32;40m{date} reformat done!\033[0m')