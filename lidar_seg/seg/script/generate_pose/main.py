import os
import os.path as osp
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
# from extract_fs import ExtractFs
# import cv2
from estimate_pose import ICP, TLG, EstimatePose
from scipy.spatial.transform import Rotation as Rot


class LastFrame:
    def __init__(self) -> None:
        self.last_stamp = None
        self.last_pole = None
        self.last_ground = None
        self.last_pc = None
        self.last_label = None
        self.last_R = None
        self.last_t = None
        self.R_sum = np.identity(3, np.float32)
        self.t_sum = np.zeros((3, 1), np.float32)
        a = osp.dirname(data_dir)
        self.odom_file = open(osp.join(a, 'odomtry.txt'), 'w')

    def __del__(self) -> None:
        self.odom_file.close()

    @staticmethod
    def one_update(str, **kwarg):
        if str in kwarg:
            a = kwarg[str]
        else:
            a = None
        return a

    def update(self, **kwarg):
        self.last_stamp = LastFrame.one_update('stamp', **kwarg)
        self.last_pole = LastFrame.one_update('pole', **kwarg)
        self.last_ground = LastFrame.one_update('ground', **kwarg)
        self.last_pc = LastFrame.one_update('pc', **kwarg)
        self.last_label = LastFrame.one_update('label', **kwarg)
        self.last_t = LastFrame.one_update('t', **kwarg)
        self.last_R = LastFrame.one_update('R', **kwarg)
        if "t" in kwarg:
            self.t_sum += self.last_t
        else:
            self.t_sum = np.zeros((3, 1), np.float32)
        if "R" in kwarg:
            self.R_sum = self.R_sum @ self.last_R.T
            q = Rot.from_matrix(self.R_sum).as_quat()
            self.odom_file.write(f"{self.last_stamp} {self.last_stamp} {-self.t_sum[0][0]}"
            f" {-self.t_sum[1][0]} {-self.t_sum[2][0]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
            self.odom_file.flush()
        else:
            self.R_sum = np.identity(3, np.float32)
            self.odom_file.write(f"{self.last_stamp} -1 0 0 0 0 0 0 1\n")
            self.odom_file.flush()



data_dir = '/data/luminar_seg/single_seg_0818/pcd_dir'
data_paths = sorted(os.listdir(data_dir))

rospy.init_node('talker_p')
pub = rospy.Publisher('pc', PointCloud2, queue_size=10)
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

extract_fs = EstimatePose()
last_frame = LastFrame()

for i, task in enumerate(data_paths):
    timestamp = task[:20]
    data_path = os.path.join(data_dir, task)
    
    pc = np.fromfile(data_path, np.float32).reshape(-1, 3)
    label_path = data_path.replace('pcd_dir', 'ann_dir')
    label = np.fromfile(label_path, np.uint8)
    pole = pc[label == 7, :]
    pole = extract_fs.GetClusterCenter(pole)
    ground = pc[label == 0, :]
    if pole.shape[0] < 3 or ground.shape[0] < 3:
        last_frame.update(stamp=timestamp)
        continue
    if last_frame.last_stamp is None or last_frame.last_ground is None:
        last_frame.update(stamp=timestamp, pole=pole,
        ground=ground, pc=pc, label=label)
        continue
    if float(timestamp) - float(last_frame.last_stamp) > 1:
        last_frame.update(stamp=timestamp, pole=pole,
        ground=ground, pc=pc, label=label)
        continue

    r_3d, z = TLG(last_frame.last_ground, ground)
    last_pole = last_frame.last_pole @ r_3d
    r_2d, t, dist = ICP(last_pole, pole, None, None)
    if dist > 1:
        last_frame.update(stamp=timestamp, pole=pole,
        ground=ground, pc=pc, label=label)
        continue
    R = r_2d@r_3d.T
    t[2] = z
    
    a_points = (R @ last_frame.last_pc.T + t).T
    a_points = np.c_[a_points, np.ones(a_points.shape[0]) * 18]
    # a_pole = (R @ last_pole.T + t).T
    # a_pole = np.concatenate((a_pole, np.ones((a_pole.shape[0], 1)) * 7), axis=1)
    points = np.concatenate((pc, label.reshape(-1, 1)), axis=1)
    points = np.concatenate((points, a_points), axis=0)
    msg.row_step = msg.point_step * points.shape[0]
    msg.width = points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points, np.float32).tostring()
    pub.publish(msg)
    print(f"publihs frame {i}, forward {np.linalg.norm(t[:2])} meters!!")
    last_frame.update(stamp=timestamp, pole=pole, ground=ground,
    pc=pc, label=label, R=R, t=t)
    
    # cv2.imshow("img", img)
    # cv2.waitKey()
# cv2.destroyAllWindows()
        