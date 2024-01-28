#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np

def talker():

    pub = rospy.Publisher('pointcloud_topic', PointCloud2, queue_size=5)
    rospy.init_node('pointcloud_publisher_node', anonymous=True)
    rate = rospy.Rate(1)

    points=np.array([[225.0, -71.0, 819.8],[237.0, -24.0, 816.0],[254.0, -82.0, 772.3]])

    while not rospy.is_shutdown():

        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "livox_frame"

        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(points, np.float32).tostring()

        pub.publish(msg)
        print("published...")
        rate.sleep()

if __name__ == '__main__':

    talker()