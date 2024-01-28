#!/usr/bin/env python3
# _*_ coding: UTF-8 _*_
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np


def talker(points_pub, points):
    # while not rospy.is_shutdown():
    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = 'velodyne'

    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('ring', 16, PointField.UINT16, 1)]

    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points, np.float32).tobytes()

    points_pub.publish(msg)
    print('pubilshed...')


if __name__ == '__main__':
    talker()
