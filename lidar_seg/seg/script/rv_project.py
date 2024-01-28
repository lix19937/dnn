# coding=utf-8

import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R
from util import color_dict, color_float, read_ascii_pcd


data_dir = '/data/rslidar/highway/lidar'
# data_dir = '/data/seg_data/ori_data/label_20220926/lidar'
pcd_list = sorted(os.listdir(data_dir))
rospy.init_node('talker_p')
pub =rospy.Publisher('pc', PointCloud2, queue_size=10)
# pub1 =rospy.Publisher('pc1', PointCloud2, queue_size=10)
msg = PointCloud2()
msg.header.stamp = rospy.Time().now()
msg.header.frame_id = "base_link"
    
msg.fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('r', 12, PointField.FLOAT32, 1),
    PointField('g', 16, PointField.FLOAT32, 1),
    PointField('b', 20, PointField.FLOAT32, 1),
    PointField('intensity', 24, PointField.FLOAT32, 1)]
msg.is_bigendian = False
msg.point_step = 28
msg.height = 1

img_scale=(192, 1024)
fov_angle=(-16, 8)
horizon_angle=(-64, 64)
fov_up = fov_angle[1] / 180.0 * np.pi  # field of view up in rad
fov_down = fov_angle[0] / 180.0 * np.pi  # field of view down in rad
fov = fov_up - fov_down  # get field of view total in rad
fov_left = horizon_angle[0] / 180.0 * np.pi
horizon_fov = horizon_angle[1] / 180.0 * np.pi - fov_left

for pcd in pcd_list:
    pcd_path = os.path.join(data_dir, pcd)
    # if not os.path.exists(task_path):
    #     continue
    # pcs = open(task_path, 'r').readlines()
    # for data_path in pcs:
    pc = read_ascii_pcd(pcd_path)
    # pc = (np.matmul(rotate.T, pc.T) - translation.reshape(3, -1)).T.astype(np.float32)
    scan_x = pc.pc_data['x']
    scan_y = pc.pc_data['y']
    scan_z = pc.pc_data['z']
    scan_i = pc.pc_data['intensity']
    depth = np.sqrt(scan_x ** 2 + scan_y ** 2)
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    proj_y = ((yaw - fov_left) / horizon_fov * img_scale[1]).astype(np.long)  # in [0.0, 1.0]
    proj_x = (img_scale[0] - (pitch - fov_down) / fov * img_scale[0]).astype(np.long)  # in [0.0, 1.0]
    label_path = pcd_path.replace('/lidar/', '/label_bin/')[:-4] + '.bin'
    if not os.path.exists(label_path):
        print(label_path)
        continue
    label = np.fromfile(label_path, np.uint8)
    img = np.zeros(img_scale + (3,), np.uint8)
    rv = np.zeros(img_scale, np.float32)
    idx = np.zeros(img_scale, np.long)
    mask = np.ones(len(proj_x), np.uint8)
    for i in range(len(proj_x)):
        if proj_x[i] < 0 or proj_x[i] >= img_scale[0] or \
            proj_y[i] < 0 or proj_y[i] >= img_scale[1]:
            mask[i] = 100
            continue
        img[proj_x[i], proj_y[i]] = color_dict[label[i]][::-1]
        rv[proj_x[i], proj_y[i]] = depth[i]
        idx[proj_x[i], proj_y[i]] = i
    ct = np.count_nonzero(rv)
    print(f'pc rate: {ct / i}, img rate: {ct / (64 * 1024)}')
    npc = []
    for r in range(rv.shape[0]):
        for c in range(rv.shape[1]):
            if rv[r, c] > 0.1:
                t_pitch = (img_scale[0] - r) * fov / img_scale[0] + fov_down
                t_yaw = -(c * horizon_fov / img_scale[1] + fov_left)
                z = rv[r, c] * np.math.sin(t_pitch)
                x = rv[r, c] * np.math.cos(t_pitch) * np.math.cos(t_yaw)
                y = rv[r, c] * np.math.cos(t_pitch) * np.math.sin(t_yaw)
                npc.append([x, y, z, *color_float[label[idx[r, c]]], label[idx[r, c]]])
    points = np.array(npc, np.float32)
    # pc = pc[npc, :]
    # label = label[npc]
    # length = len(label)
    # tpoints = np.concatenate((pc, label.reshape(-1, 1)), axis=1)
    msg.row_step = msg.point_step * points.shape[0]
    msg.width = points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points, np.float32).tostring()
    pub.publish(msg)
    # msg.row_step = msg.point_step * tpoints.shape[0]
    # msg.width = tpoints.shape[0]
    # msg.data = np.asarray(tpoints, np.float32).tostring()
    # pub1.publish(msg)
    cv2.imshow('img', img)
    cv2.imshow('rv', rv/20)
    if cv2.waitKey() == 27:
        exit()
