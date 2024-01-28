import os
import os.path as osp
import cv2
import numpy as np
from ftp import MyFtp
import shutil
import pcd_io

label_name2 = ('road', 'sidewalk', 'terrain', 'lane', 'fence', 'wall',
           'building', 'suspension', 'pole', 'vegetation', 'traffic sign', 'traffic cone', 'object_other',
           'Traffic participants', 'wheel stop', 'barrier gate', 'reflection',
           'mist', 'others',)

label_name1 = ('road', 'sidewalk', 'terrain', 'fence', 'wall',
           'building', 'suspension', 'pole', 'vegetation', 'traffic sign', 'traffic cone', 'object_other',
           'vehicle', 'non-motor vehicle', 'person', 'wheel stop', 'barrier gate', 'reflection',
           'mist', 'others',)

label_name = ('路面', '人行道', '泥地', '交通标识线', '护栏', '墙体',
           '建筑', '高空物', '支撑杆', '低空植被', '交通牌', '交通锥', '其他目标',
           '车或者人', '轮挡', '关闸', '反射噪声',
           '水雾沙尘', '未知目标', '红绿灯', '挡板')

color_dict = [[255, 0, 0], [255, 127, 80], [218, 97, 17], [255, 192, 203],
           [255, 255, 0], [0, 200, 200], [128, 128, 0], [200, 200, 200],
           [107, 142, 35], [0, 255, 127], [152, 251, 152], [100, 60, 255], [142, 0, 252],
           [0, 0, 255], [60, 179, 113], [255, 215, 0], [0, 60, 100],
           [0, 80, 100], [72, 51, 159], [119, 11, 32], [128, 64, 128]]

color_float = [[1, 0, 0], [1, 0.5, 0.3], [0.85, 0.38, 0.07], [1, 0.75, 0.8],
           [1, 1, 0], [0, 0.78, 0.78], [0.5, 0.5, 0], [0.8, 0.8, 0.8],
           [0.42, 0.56, 0.14], [0, 1, 0.5], [0.6, 1, 0.6], [0.39, 0.24, 1], [0.56, 0, 1],
           [0, 0, 1], [0.24, 0.7, 0.44], [1, 0.84, 0], [0, 0.24, 0.39],
           [0, 0.31, 0.39], [0.28, 0.2, 0.62], [0.47, 0.04, 0.13], [0.5, 0.25, 0.5]]

def is_float(text: str) -> bool:
    try:
        float(text)
    except:
        return False
    return True

def convert_9_to_6(file_name):
    return format(float(file_name), '.6f')

def read_ascii_pcd(path, filter=False):
    pcd = pcd_io.point_cloud_from_path(path)
    if filter:
        filtered_data = pcd.pc_data[pcd.pc_data['x'] > 0.1]
    else:
        filtered_data = pcd.pc_data
    pc_size = filtered_data.shape[0]
    # head = pcd.get_metadata()
    mydtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
    new_pc = np.zeros((pc_size, ), dtype=mydtype)
    new_pc['x'] = filtered_data['x']
    new_pc['y'] = filtered_data['y']
    new_pc['z'] = filtered_data['z']
    new_pc['intensity'] = filtered_data['intensity']
    meta = {'version':0.7,'fields':['x', 'y', 'z', 'intensity'],'size':[4, 4, 4, 4],
    'type':['F', 'F', 'F', 'F'], 'count':[1,1,1,1], 'width':pc_size,
    'height': 1, 'viewpoint':[0,0,0,1,0,0,0], 'points':pc_size, 'data':'ascii'}
    return pcd_io.PointCloud(meta, new_pc)

def create_palette(root_dir):
    palette_dir = os.path.join(root_dir, 'palette_luminar')
    if not os.path.exists(palette_dir):
        os.makedirs(palette_dir)
        for i, name in enumerate(label_name):
            color = color_dict[i]
            img = np.zeros((100, 100, 3), np.uint8)
            img[:, :, 0] = color[2]
            img[:, :, 1] = color[1]
            img[:, :, 2] = color[0]
            cv2.imwrite(os.path.join(palette_dir, f'{i}_{name}.png'), img)

def put_data(out_dir):
    ftp = MyFtp(username="igs",password="igs",host='10.95.61.167',port=22)
    _, sftp = ftp.sftp_connect()
    if ftp.sftp_upload_dir(sftp, out_dir):
        print(f'upload {out_dir} finished!')
    sftp.close()

def path_change(path: str) -> str:
    dirs = path.split('/')
    idx = dirs.index('Lidar_OD_SEG')
    label_task = dirs[idx + 1]
    dirs.insert(idx+1, '数据送标')
    dirs.insert(idx+2, label_task[6:10])
    dirs.insert(idx+3, label_task[6:12])
    return '/'.join(dirs)

def find_jpg(out_dir):
    import time
    from tqdm import tqdm
    file_list = os.listdir(osp.join(out_dir, 'pcd_dir'))
    file_list = sorted([_.split('_')[0] for _ in file_list])
    tgt_dir = osp.join(out_dir, 'jpg_dir')
    last_date = ''
    for file in tqdm(file_list):
        date = time.strftime('%Y%m%d', time.localtime(float(file)))
        if date != last_date:
            t_list = open(f'/data/seg_align/{date}/train_pcd.txt', 'r').readlines()
            t_list = [path_change(_) for _ in t_list]
            t_list = dict(zip([_.split('/')[-1].strip()[:-4] for _ in t_list], t_list))
            last_date = date
        if file in t_list:
            shutil.copy(t_list[file].replace('.pcd\n', '.jpg'), tgt_dir)
        else:
            print(f'DO NOT FIND {file} jpg file!')

def remove_same():
    root_dir = '/data/seg_data/ori_data'
    for date in sorted(os.listdir(root_dir)):
        file_path = osp.join(root_dir, date, 'filter.txt')
        if not osp.exists(file_path):
            continue
        data = open(file_path, 'r').readlines()
        # for scene in scenes:
        # train_path = osp.join(root_dir, weather, scene, 'train.txt')
        # if not osp.exists(train_path):
        #     continue
        # data = open(train_path, 'r').readlines()
        print(date, len(data))
        # data = [_ for _ in data]
        data = list(set(data))
        print(len(data))
        data.sort()
        f = open(file_path, 'w')
        f.writelines(data)
        f.close()
        # [f.write(_ + '\n') for _ in data]

if __name__ == "__main__":
    # find_jpg('/data/luminar_seg/single_seg_0507')
    remove_same()