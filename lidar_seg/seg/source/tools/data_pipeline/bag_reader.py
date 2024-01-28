import rosbag
import math
import os
import json
# import sys
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from datasets.sample.lidar_seg import LidarSeg
from datasets.dataset.luminar_seg import LuminarSeg

class Dataset(LuminarSeg, LidarSeg):
        def __init__(self, opt, split) -> None:
            super().__init__(opt, split)
            LidarSeg.__init__(self)

def read_imu(bag_file):
    bag = rosbag.Bag(bag_file, 'r')
    
    bag_data = bag.read_messages('/Saic_IPS/ips_base')
    if not os.path.exists('imu'):
        os.makedirs('imu')
    for topic, msg, t in bag_data:
        data = {}
        data['stamp'] = msg.header.stamp.to_nsec()
        data['velo'] = math.hypot(msg.ips.VelN[0], msg.ips.VelN[1])
        data['yawrate'] = math.radians(msg.dr.AngleRate_FLU[2])
        with open('imu/{}.json'.format(data['stamp']),'w') as f:
            json.dump(data, f)

def read_one_point_cloud(opt, bag_file):
    # bag = rosbag.Bag(bag_file, 'r')
    lidar_od = Dataset(opt, 'val')
    # bag_data = bag.read_messages('/luminar_driver/luminar_points')
    # for _, msg, _ in bag_data:
    #     lidar = pc2.read_points(msg)
    #     points = np.array(list(lidar))[:, :4].astype(np.float32)
    #     result = lidar_od.project_to_bev(points)
    #     return result
    points = np.fromfile('exp/cloud.bin', np.float32).reshape(-1, 4)
    result = lidar_od.project_to_bev(points)
    return result

read_imu('/data/2021-12-16-10-35-00_8.bag')
