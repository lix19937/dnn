import os
import numpy as np
import struct
import open3d
import cv2
import struct
import time
import matplotlib.pyplot as plt
import math
import json

def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R

def main():

    with open('/home/igs/train_small/json/1632282278531605000.json', 'r') as f:
        lidar_one_frame = json.load(f)
    lidar_one_frame = lidar_one_frame['cloud']

    #arr=np.array(list(lidar_one_frame))
    a = [list(i.values()) for i in lidar_one_frame]
    arr = np.array(a)
    vis = open3d.visualization.Visualizer()
    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(arr)
    #pcd.points = open3d.utility.Vector3dVector(lidar_one_frame)
    vis.create_window(window_name='lidar_od', width=512, height=512)
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd)

    with open('/home/igs/Desktop/nn-detection-centernet/source/result.json', 'r') as f:
        obj = json.load(f)#len(obj)
    for obj_index in range(len(obj)):

        R = rot_y(0)
        l = abs(obj[obj_index][2] - obj[obj_index][0])
        w = abs(obj[obj_index][3] - obj[obj_index][1])
        h = 2
        #h, w, l = obj[obj_index].dimensions[0], obj[obj_index].dimensions[1], obj[obj_index].dimensions[2]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        z = [0, 0, 0, 0, -h, -h, -h, -h]
        # y = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        y = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corner_3d = np.vstack([x, y, z])
        corner_3d = np.dot(R, corner_3d)

        corner_3d[0, :] += (obj[obj_index][2] + obj[obj_index][0])/2.0
        corner_3d[1, :] += (obj[obj_index][3] + obj[obj_index][1])/2.0
        corner_3d[2, :] += 0
        corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
        corner_3d[-1][-1] = 1

        position = corner_3d
        points_box = np.transpose(position)
        s = np.array(points_box)
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])
        colors = np.array([[1., 0., 0.] for j in range(len(lines_box))])
        line_set = open3d.open3d.geometry.LineSet()

        line_set.points = open3d.utility.Vector3dVector(s[:, 0:3].reshape(-1, 3))
        line_set.lines = open3d.utility.Vector2iVector(lines_box)
        line_set.colors = open3d.utility.Vector3dVector(colors)

        vis.add_geometry(line_set)
        vis.get_render_option().line_width = 5.0
        #vis.update_geometry(line_set)
       # vis.get_render_option().background_color = np.asarray([1, 1, 1])
        # vis.get_render_option().load_from_json('renderoption_1.json')
        # param = o3d.io.read_pinhole_camera_parameters('BV.json')

   # vis.update_geometry()
   # vis.poll_events()
   # vis.update_renderer()
    vis.update_geometry(line_set)
    print("s")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

