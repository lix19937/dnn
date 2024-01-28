import pcd_io
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

def convert_bin_to_pcd(bin_path, pcd_path):
    xyz = np.fromfile(bin_path, np.float32).reshape(-1, 3)
    pc_size = xyz.shape[0]
    dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
    new_pc = np.zeros((pc_size, ), dtype=dtype)
    new_pc['x'] = xyz[:, 0]
    new_pc['y'] = xyz[:, 1]
    new_pc['z'] = xyz[:, 2]
    new_pc['intensity'] = np.zeros(pc_size, np.float32)
    meta = {'version':0.7,'fields':['x', 'y', 'z', 'intensity'],'size':[4, 4, 4, 4],
    'type':['F', 'F', 'F', 'F'], 'count':[1,1,1,1], 'width':pc_size,
    'height': 1, 'viewpoint':[0,0,0,1,0,0,0], 'points':pc_size, 'data':'ascii'}
    pcd = pcd_io.PointCloud(meta, new_pc)
    pcd.save_pcd(pcd_path)

def main():
    pcd_dir = osp.join(root_dir, 'pcd_dir')
    ann_dir = pcd_dir.replace('pcd_dir', 'ann_dir')
    ori_pcd = pcd_dir.replace('pcd_dir', 'ori_pcd')
    os.makedirs(ori_pcd, exist_ok=True)
    data_list = sorted(os.listdir(pcd_dir))
    for data in tqdm(data_list):
        # data = data_list[i]
        xyz = np.fromfile(osp.join(pcd_dir, data), np.float32).reshape(-1, 3)
        label = np.fromfile(osp.join(ann_dir, data), np.uint8)
        pc_size = xyz.shape[0]
        dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u1'), ('intensity', 'f4')])
        new_pc = np.zeros((pc_size, ), dtype=dtype)
        new_pc['x'] = xyz[:, 0]
        new_pc['y'] = xyz[:, 1]
        new_pc['z'] = xyz[:, 2]
        new_pc['intensity'] = np.zeros(label.shape, np.float32)
        new_pc['label'] = label
        meta = {'version':0.7,'fields':['x', 'y', 'z', 'label', 'intensity'],'size':[4, 4, 4, 1, 4],
        'type':['F', 'F', 'F', 'U', 'F'], 'count':[1,1,1,1,1], 'width':pc_size,
        'height': 1, 'viewpoint':[0,0,0,1,0,0,0], 'points':pc_size, 'data':'ascii'}
        pcd = pcd_io.PointCloud(meta, new_pc)
        pcd.save_pcd(osp.join(ori_pcd, f'{data[:-4]}.pcd'))

if __name__ == '__main__':
    root_dir = '/data/luminar_seg/luminar_seg_0415'
    main()
