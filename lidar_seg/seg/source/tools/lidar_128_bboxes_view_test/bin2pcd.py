import numpy as np
import os

def bin2pcd(bin_url, pcd_url):
    points = np.fromfile(bin_url, dtype="float32").reshape((-1, 4))
    handle = open(pcd_url, 'a')
    point_num = points.shape[0]
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA binary_compressed')


    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()

def main():
    bin_root_path = '/data/data-data/data/lidar128_od_3.2w/clouds/0026/bin/'
    pcd_save_path = '/data/lidar_128_dt_annos/gt_annos/pcd/'
    bin_file_lists = os.listdir(bin_root_path)
    for i in range(len(bin_file_lists)):
        bin_file_name = bin_file_lists[i]
        pcd_file_name = bin_file_name.split('.')[0]+'.pcd'
        pcd_url = pcd_save_path + pcd_file_name
        bin_url = bin_root_path + bin_file_name
        print(f'########starting convert {bin_file_name} to {pcd_file_name} {i} ###########')
        bin2pcd(bin_url, pcd_url)
    print('#######save ending ###')

if __name__ == '__main__':
    main()
