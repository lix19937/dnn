import rospy
import pcl
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

cloud = pcl.load('/home/igs/Downloads/1638847233642824000.pcd')
cloud = list(cloud)
cloud = np.array(cloud, dtype=np.float32)

rospy.init_node('pcd2rviz', anonymous=True)

points_puber = rospy.Publisher(
    "/rs_lidar128_transformed", PointCloud2, queue_size=10)
msg = PointCloud2()
msg.header.stamp = rospy.Time().now()
msg.header.frame_id = 'rslidar_128'
msg.height = 1
msg.width = len(cloud)
msg.fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
]
msg.is_bigendian = False
msg.point_step = 12
msg.row_step = msg.point_step * cloud.shape[0]
msg.is_dense = False
msg.data = np.asfarray(cloud, np.float32).tostring()

rate = rospy.Rate(10.0)
while not rospy.is_shutdown():
    points_puber.publish(msg)
    rate.sleep()
