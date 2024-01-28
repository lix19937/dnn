# coding=utf-8

# Copes with wrong collections of labels and clouds.

import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import matplotlib.pyplot as plt
from convert_luminar_seg import color_dict
from scipy.spatial.transform import Rotation as R


data_dir = '/data/luminar_seg/single_seg_0806/pcd_dir/'
data_paths = os.listdir(data_dir)
data_paths.sort()
rospy.init_node('talker_p')
pub =rospy.Publisher('pc', PointCloud2, queue_size=10)
msg = PointCloud2()
msg.header.stamp = rospy.Time().now()
msg.header.frame_id = "base_link"
    
msg.fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('label', 12, PointField.FLOAT32, 1)]
msg.is_bigendian = False
msg.point_step = 16
msg.height = 1

img_scale=(64, 1024)
# fov_angle=(-16, 8)
horizon_angle=(-64, 64)
# fov_up = fov_angle[1] / 180.0 * np.pi  # field of view up in rad
# fov_down = fov_angle[0] / 180.0 * np.pi  # field of view down in rad
# fov = fov_up - fov_down  # get field of view total in rad
fov_left = horizon_angle[0] / 180.0 * np.pi
horizon_fov = horizon_angle[1] / 180.0 * np.pi - fov_left

quat_xyzw = [0, 0, 0, 1]
translation = np.array([1.5, 0, 2.0], np.float32)
rotate = R.from_quat(quat_xyzw).as_matrix()
for _ in data_paths:
    data_path = os.path.join(data_dir, _)
    pc = np.fromfile(data_path, np.float32).reshape(-1, 3)
    pc = (np.matmul(rotate.T, pc.T) - translation.reshape(3, -1)).T.astype(np.float32)
    label_path = data_path.replace('pcd_dir', 'ann_dir')
    label = np.fromfile(label_path, np.uint8)
    t = [1.5, 0, 1.5]
    dis = np.sqrt((pc[:, 0]  - t[0]) ** 2 + pc[:, 1] ** 2)
    z = pc[:, 2] - t[2]
    pitch = np.arcsin(z / dis)
    yaw = -np.arctan2(pc[:, 1], pc[:, 0])
    last = yaw[0]
    # pc_line = []
    # tmp_line = pc[0, :]
    line_seq = 0
    lines = np.zeros((pc.shape[0],), np.int)
    # t = np.zeros(32, np.int)
    for i, y in enumerate(yaw[1:]):
        if abs(y - last) > 0.8:
            # pc_line.append(tmp_line.reshape(-1, 3))
            # tmp_line = pc[i + 1, :]
            # t[line_seq] = i
            line_seq = line_seq + 1
        else:
            lines[i] = 2 * line_seq if i % 2 == 0 else 2 * line_seq + 1
        last = y
    # t[line_seq] = i
    # pc_line.append(tmp_line.reshape(-1, 3))

    # plt.hist(lines, bins=64)
    # print(max(lines))
    # plt.show()
    scan_x = pc[:, 0]
    scan_y = pc[:, 1]
    scan_z = pc[:, 2]
    depth = np.linalg.norm(pc, 2, axis=1)
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    proj_y = ((yaw - fov_left) / horizon_fov * img_scale[1]).astype(np.long)  # in [0.0, 1.0]
    # proj_x = (img_scale[0] - (pitch - fov_down) / fov * img_scale[0]).astype(np.long)  # in [0.0, 1.0]
    label_path = data_path.replace('pcd_dir', 'ann_dir')
    label = np.fromfile(label_path, np.uint8)
    img = np.zeros(img_scale + (3,), np.uint8)
    rv = np.zeros(img_scale, np.float32)
    # idx = np.zeros(img_scale, np.long)
    mask = np.ones(len(proj_y), np.bool)
    for i in range(len(proj_y)):
        if proj_y[i] < 0 or proj_y[i] >= img_scale[1]:
            mask[i] = False
            continue
        img[min(lines[i], 63), proj_y[i]] = color_dict[label[i]][::-1]
        rv[min(lines[i], 63), proj_y[i]] = depth[i]
        # idx[proj_x[i], lines[i]] = i
    
    # rv = np.sum(img, axis=-1)
    ct = np.count_nonzero(rv)
    print(f'pc rate: {ct / i}, img rate: {ct / (64 * 1024)}')
    
    length = len(label)
    points = np.concatenate((pc, label.reshape(-1, 1)), axis=1)
    points = points[mask, :]

    msg.row_step = msg.point_step * points.shape[0]
    msg.width = points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points, np.float32).tostring()
    pub.publish(msg)

    cv2.imshow('img', img)
    cv2.imshow('rv', rv/20)
    if cv2.waitKey() == 27:
        exit()

