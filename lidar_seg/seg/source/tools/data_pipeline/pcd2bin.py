# coding=utf-8

# import pcl
import open3d
import os
import numpy as np
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
# e.g. --input_dir xxx/clouds_pcd/
parser.add_argument('--input_dir', required=True, type=str)
args = parser.parse_args()


data_dir = args.input_dir
all_folders = [os.path.join(data_dir, one_folder)
               for one_folder in os.listdir(data_dir)]
valid_folders = []
for one in all_folders:
    if os.path.isdir(one):
        valid_folders.append(one)
valid_folders.sort()
print(valid_folders)


def process_one_folder(target_folder):
    bin_dir = target_folder.replace('clouds_pcd', 'clouds')
    bin_dir = os.path.join(bin_dir, 'bin')
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    # pcd_dir = os.path.join(target_folder, 'pcd')
    pcd_dir = target_folder
    for one_pcd in os.listdir(pcd_dir):
        if '.pcd' not in one_pcd:
            continue
        pcd_file = os.path.join(pcd_dir, one_pcd)

        # cloud = pcl.load(pcd_file)
        # cloud = list(cloud)
        # cloud = np.array(cloud, dtype=np.float32)
        cloud = open3d.io.read_point_cloud(pcd_file)
        cloud = np.asarray(cloud.points, dtype=np.float32)

        time = pcd_file.split('/')[-1].replace('.pcd', '')
        print(target_folder, time)
        bin_file = os.path.join(bin_dir, time+'.bin')
        cloud.tofile(bin_file)


pool = Pool(processes=len(valid_folders)+1)
for idx in range(len(valid_folders)):
    pool.apply_async(process_one_folder, args=(valid_folders[idx],))

pool.close()
pool.join()
