import os
import os.path as osp
import sys
import numpy as np
import open3d as o3d
from view_lidar import color_dict
from pt_vis import Window, Visualization, ImgThread
from tkinter import *
import time

pc_indices = []
pc_list = []
filter_name = dict()

def run(index, pc_name, pc, jpg_path):
    global pc_indices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc[0])
    point_cloud.colors = o3d.utility.Vector3dVector(pc[1])
    print("receive point size:", pc[0].shape[0])
    vis = Visualization(str(index) + ' ' + pc_name, point_cloud)
    p = ImgThread(jpg_path)
    p.start()
    time.sleep(1)
    vis.vis.run()
    picked = vis.vis.get_picked_points()
    pc_indices = [_.index for _ in picked]

def pub_pc(index):
    pc = pc_list[index]
    pc_path = os.path.join(pcd_dir, pc)
    ann_path = os.path.join(ann_dir, pc)
    jpg_path = os.path.join(jpg_dir, pc.split('_')[0] + '.jpg')
    points = np.fromfile(pc_path, np.float32).reshape(-1, 3)
    gt_seg_label = np.fromfile(ann_path, np.uint8).reshape(-1, 1)
    rgb = np.ones((points.shape[0], 3), np.float32)
    for t in range(points.shape[0]):
        rgb[t, :] = np.asarray(color_dict[gt_seg_label[t, 0]], np.float32) / 255
    try:
        truth = int(myWindow.entry4.get())
    except:
        truth = -1
    label_valid = ((gt_seg_label != truth)).reshape(-1)
    pc_size = points.shape[0]
    indices = np.array(range(pc_size))
    indices = indices[label_valid]
    run(index, pc, (points[label_valid, :], rgb[label_valid, :]), jpg_path)
    global pc_indices
    pc_indices = indices[pc_indices]

def up(event):
    idx = int(myWindow.entry2.get())
    if idx <= 0:
        return
    idx -= 1
    while pc_list[idx] in filter_name:
        idx -= 1
        if idx < 0:
            return
    myWindow.entry2.delete(0, END)
    myWindow.entry2.insert(END, str(idx))
    myWindow.entry1.delete(0, END)
    text = pc_list[idx]
    myWindow.entry1.insert(END, text)
    filter_data()

def down(event):
    idx = int(myWindow.entry2.get())
    if idx >= length - 1:
        return
    idx += 1
    while pc_list[idx] in filter_name:
        idx += 1
        if idx > length - 1:
            return
    myWindow.entry2.delete(0, END)
    myWindow.entry2.insert(END, str(idx))
    myWindow.entry1.delete(0, END)
    text = pc_list[idx]
    myWindow.entry1.insert(END, text)
    filter_data()

def enter(event):
    idx = int(myWindow.entry2.get())
    myWindow.entry1.delete(0, END)
    text = pc_list[idx]
    myWindow.entry1.insert(END, text)
    pub_pc(idx)

def filter_data():
    idx = int(myWindow.entry2.get())
    wrong = int(myWindow.entry3.get())
    pc = pc_list[idx]
    pc_path = os.path.join(pcd_dir, pc)
    ann_path = os.path.join(ann_dir, pc)
    jpg_path = os.path.join(jpg_dir, pc.split('_')[0] + '.jpg')
    points = np.fromfile(pc_path, np.float32).reshape(-1, 3)
    gt_seg_label = np.fromfile(ann_path, np.uint8).reshape(-1, 1)
    rgb = np.ones((points.shape[0], 3), np.float32)
    for t in range(points.shape[0]):
        rgb[t, :] = np.asarray(color_dict[gt_seg_label[t, 0]], np.float32) / 255
    pc_size = points.shape[0]
    indices = np.array(range(pc_size))
    label_valid = ((gt_seg_label == wrong)).reshape(-1)
    indices = indices[label_valid]
    if indices.shape[0] == 0:
        print('no point!')
        return
    run(idx, pc, (points[label_valid, :], rgb[label_valid, :]), jpg_path)
    global pc_indices
    pc_indices = indices[pc_indices]

def change_label():
    idx = int(myWindow.entry2.get())
    if len(pc_indices) > 0:
        wrong = int(myWindow.entry3.get())
        truth = int(myWindow.entry4.get())
        pc = pc_list[idx]
        ann_path = os.path.join(ann_dir, pc)
        gt_seg_label = np.fromfile(ann_path, np.uint8).reshape(-1, 1)
        change_indices = [_ for _ in pc_indices if gt_seg_label[_] == wrong]
        gt_seg_label[change_indices] = truth
        gt_seg_label.tofile(ann_path)
        # filter_data()
        pub_pc(idx)

if __name__ == "__main__":
    data_dir = sys.argv[1]
    pcd_dir = osp.join(data_dir, 'pcd_dir')
    ann_dir = osp.join(data_dir, 'ann_dir')
    jpg_dir = osp.join(data_dir, 'jpg_dir')
    pc_list = sorted(os.listdir(pcd_dir))

    length = len(pc_list)
    myWindow = Window(length, filter_data, change_label, up, down, enter)
    if length == 0:
        exit(-1)
    
    last_idx = 0
    myWindow.entry2.insert(END, last_idx)
    
    myWindow.run()
