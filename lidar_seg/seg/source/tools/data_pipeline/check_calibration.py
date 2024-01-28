from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

from matplotlib import pyplot as plt
import numpy as np
import json

cloud_path = '/home/igs/Downloads/Lidar_OD_20211210_5k/LidarOD-20211210/clouds/市区/1637737709.000105000.bin'
label_path = '/home/igs/Downloads/Lidar_OD_20211210_5k/LidarOD-20211210/labels/市区/1637737709.000105000.json'

yaw = -1.16944071e-02
rot_mat = np.array([
    [9.99924839e-01, -1.13726659e-02, -4.58624261e-03],
    [1.14255212e-02, 9.99866664e-01, 1.16680861e-02],
    [4.45293356e-03, -1.17196087e-02, 9.99921381e-01]], dtype=np.float32)
trans_mat = np.array(
    [1.66035831e+00, 3.09985187e-02, 1.61256278e+00],
    dtype=np.float32
)


with open(label_path, 'r') as f:
    data = json.load(f)
objects = data['Objects']
parsed_results = []
for one_obj in objects:
    center_lidar = np.array([
        float(one_obj['center'][0]), float(one_obj['center'][1]), float(one_obj['center'][2])],
        dtype=np.float32)
    center_ego = center_lidar @ rot_mat.T + trans_mat
    parsed_obj = {
        'center': center_ego.tolist(),
        'heading': float(one_obj['heading']) + yaw,
        'dim': [float(one_obj['dim'][0]), float(one_obj['dim'][1]), float(one_obj['dim'][2])]}

    parsed_results.append(parsed_obj)

lidar_one_frame = np.fromfile(cloud_path, dtype=np.float32)
lidar_one_frame = lidar_one_frame.reshape(-1, 3)
lidar_one_frame = lidar_one_frame @ rot_mat.T + trans_mat


cloud_points = []
for idx in range(len(lidar_one_frame)):
    p = lidar_one_frame[idx]
    if idx % 2 != 0 or np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]):
        continue
    if p[0] > 102.4 or p[0] < 0.0 or np.abs(p[1]) > 51.2:
        continue
    cloud_points.append([p[0], p[1]])
plt.figure()
plt.xlim((0, 105))
plt.ylim((-40, 40))
cloud_points = np.array(cloud_points)
plt.scatter(cloud_points[:, 0],
            cloud_points[:, 1], s=0.01, c='b')

for obj in parsed_results:
    l = obj['dim'][0]
    w = obj['dim'][1]
    corners = []
    corners.append([l/2., w/2.])
    corners.append([l/2., -w/2.])
    corners.append([-l/2., -w/2.])
    corners.append([-l/2., w/2.])
    corners.append([l/2., w/2.])
    x_c = obj['center'][0]
    y_c = obj['center'][1]
    theta = obj['heading']
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    # print('=== theta deg', theta * 180.0 / np.pi)
    for idx in range(len(corners)):
        x_tmp = cos_t * corners[idx][0] - \
            sin_t * corners[idx][1] + x_c
        y_tmp = sin_t * corners[idx][0] + \
            cos_t * corners[idx][1] + y_c
        corners[idx][0] = x_tmp
        corners[idx][1] = y_tmp
    corners = np.array(corners)
    plt.plot(corners[:, 0], corners[:, 1], c='r')
plt.savefig('test_calibration.jpg')
plt.close()
