from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import _init_paths
from models.model import create_model, load_model
from opts import opts
from datasets.sample import preprocess_cpp
from lib.utils.post_process import bev_nms
from tools.calibration import veh2cali
import numpy as np
import torch
import torch.utils.data
import rospy
import time
from sensor_msgs.msg import PointCloud2, PointField
from tools.lidar_128_bboxes_view_test.jsk_3dbox_view import pub_bb
from jsk_recognition_msgs.msg import BoundingBoxArray

import pyximport
pyximport.install()


class get_bboxes_from_model(object):
    def __init__(self,
                 lidar_type,                   # 'lidar128' or 'luminar'.
                 pub_bboxes_topic,    # pub box topic.
                 pub_points_topic,     # pub transformed cloud topic.
                 vehicle_no,                  # vehicle number for reading RT.
                 score_thr=0.5             # score threshold for obj filter.
                 ):
        assert lidar_type in ('lidar128', 'luminar')
        self.lidar_type = lidar_type
        self.sub_points_topic = \
            '/rslidar_points_128' if lidar_type == 'lidar128' else '/luminar_driver/luminar_points'
        self.pub_bboxes_topic = pub_bboxes_topic
        self.opt = opts()
        self.opt = self.opt.parse()

        self.model = self.get_self_model()
        self.model.eval()
        # Publishes  pointscloud  to  rviz.
        self.points_puber = rospy.Publisher(
            pub_points_topic, PointCloud2, queue_size=10)
        # Publishes  bboxes  to  rviz.
        self.bb_puber = rospy.Publisher(
            self.pub_bboxes_topic, BoundingBoxArray, queue_size=10)
        # Subscribes pointscloud  from  rosbag  node.
        self.suber = rospy.Subscriber(
            self.sub_points_topic, PointCloud2, self.callback, queue_size=10)
        self.socre_thr = score_thr
        self.calibration = veh2cali[vehicle_no]
        self.rot_mat = np.array(
            self.calibration['rotation_matrix'], dtype=np.float32)
        self.trans_mat = np.array(
            self.calibration['translation_matrix'], dtype=np.float32)

    def pointcloud2_to_array(self, cloud_msg):
        cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.float32)
        field_num = 4 if self.lidar_type == 'lidar128' else 10
        cloud_arr = cloud_arr.reshape(-1, field_num)[:, :3]
        return cloud_arr

    def callback(self, msg):
        assert isinstance(msg, PointCloud2)
        raw_lidar_128_points = self.pointcloud2_to_array(msg)
        raw_lidar_128_points = raw_lidar_128_points @ self.rot_mat.T + self.trans_mat
        ####convert raw_lidar_points to inputs for model######
        raw_lidar_128_points = torch.from_numpy(raw_lidar_128_points)
        model_inputs = self.convert_points_to_model_input(raw_lidar_128_points)
        model_inputs = np.expand_dims(model_inputs, axis=0)
        model_inputs = torch.from_numpy(model_inputs)
        ####detection####
        if self.opt.gpus[0] >= 0:
            self.opt.device = torch.device('cuda')
        else:
            self.opt.device = torch.device('cpu')
        with torch.no_grad():
            output = self.model(model_inputs.to(self.opt.device))
            output = output[-1]
            pre_bboxes = []
            dets_post = output['decoded'].cpu()
            dets_post = bev_nms(dets_post, self.socre_thr)
            for obj in dets_post:
                if obj[-2] < self.socre_thr:
                    continue
                l = obj[3]
                w = obj[4]
                h = obj[5]
                x_c = obj[0]
                y_c = obj[1]
                z_c = obj[2]
                theta = obj[6]
                pre_bboxes.append([x_c, y_c, z_c, l, w, h, theta])

            self.bboxes_pub(np.array(pre_bboxes))
            self.pub_points(raw_lidar_128_points.numpy())

    def get_self_model(self):
        opt = opts().set_input_info_and_heads(self.opt)
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        model = create_model(opt, 'val')
        model = load_model(model, opt.load_model)
        model = model.to(opt.device)
        return model

    def convert_points_to_model_input(self, raw_lidar_128_points):
        self.roi_x_max = self.opt.roi[3]
        self.roi_x_min = self.opt.roi[0]
        self.roi_y_max = self.opt.roi[4]
        self.roi_y_min = self.opt.roi[1]
        self.min_z = self.opt.roi[2]
        self.max_z = self.opt.roi[5]
        self.bev_res = self.opt.bev_res
        self.channels_num = 16
        self.height_res = self.opt.height_res
        model_input = preprocess_cpp.build(
            raw_lidar_128_points,
            self.channels_num,
            self.opt.input_h,
            self.opt.input_w,
            self.roi_x_min,
            self.roi_y_min,
            self.min_z,
            self.max_z,
            self.bev_res,
            self.height_res)
        model_input = model_input.numpy()
        model_input[0, :, :] /= self.roi_x_max
        model_input[1, :, :] /= self.roi_y_max
        model_input[2:, :, :] = np.log(model_input[2:, :, :] + 1.0)

        return model_input

    def bboxes_pub(self, pre_bboxes):
        pre_bboxes = pre_bboxes.tolist()
        pub_bb(self.bb_puber, pre_bboxes)

    def pub_points(self, points):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = 'rslidar_128'
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = np.asfarray(points, np.float32).tostring()
        self.points_puber.publish(msg)


if __name__ == '__main__':
    rospy.init_node('lidar_od', anonymous=True)
    get_bboxes_from_model(
        'luminar',
        'lidar_128_jsk_pub_3dbboxes',
        '/rs_lidar128_transformed',
        'v10')
    rospy.spin()

"""python3 ros_pipeline.py lidar_od --arch hourglass --down_ratio 4 --gpus 0 --num_stacks 2 --dataset lidar128 --load_model xxx.pth"""
