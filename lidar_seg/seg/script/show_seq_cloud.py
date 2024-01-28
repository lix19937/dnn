import math
import os
import os.path as osp
import shutil
import time
from tqdm import tqdm
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField

root_dir = '/data/luminar_seg/baffle_0527'
pcd_dir = osp.join(root_dir, 'pcd_dir')
last_pcd_dir = osp.join(root_dir, 'last_pcd_dir')

def get_transform(one_odom, dt):
    theta = one_odom[1] * dt
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([[cos_theta, sin_theta, one_odom[0] * dt],
                     [-sin_theta, cos_theta, 0],
                     [0, 0, 1]])

def show_seq_cloud():
    rospy.init_node('talker_p')
    #3.实例化 发布者 对象(发布话题-chatter，std_msgs.msg.String类型,队列条目个数)
    pub =rospy.Publisher('pc', PointCloud2, queue_size=10)
    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = "base_link"
    
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.height = 1
    data_list = sorted(os.listdir(pcd_dir))
    for data_path in tqdm(data_list):
        stamp = float(data_path[:20])
        data_abs_path = osp.join(pcd_dir, data_path)
        odom_path = data_abs_path.replace('pcd_dir', 'last_odom_dir')
        odom_data = np.loadtxt(odom_path)
        if odom_data[-1] == 0:
            continue
        last_data_path = data_abs_path.replace('pcd_dir', 'last_pcd_dir')
        lidar_one_frame = np.fromfile(data_abs_path, np.float32).reshape(-1, 3)
        lidar_last_frame = np.fromfile(last_data_path, np.float32).reshape(-1, 4)[:, :3]
        last_stamp = float(odom_data[0])
        tf = get_transform(odom_data[1:], stamp - last_stamp)
        tmp = np.ones(lidar_last_frame.shape, np.float32)
        tmp[:, :2] = lidar_last_frame[:, :2]
        # lidar_last_frame[:, 3] = 1
        lidar_last_frame[:, :2] = (tf @ tmp.T).T[:, :2]
        merge_frame = np.vstack((lidar_one_frame, lidar_last_frame))
        rgb = np.ones((merge_frame.shape[0], 3), np.float32)
        for t in range(lidar_one_frame.shape[0]):
            rgb[t, :] = np.asarray([1, 0, 0], np.float32)
        points = np.concatenate((merge_frame, rgb), axis=1)
        msg.row_step = msg.point_step * points.shape[0]
        msg.width = points.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(points, np.float32).tostring()
        pub.publish(msg) #发布信息到主题

show_seq_cloud()