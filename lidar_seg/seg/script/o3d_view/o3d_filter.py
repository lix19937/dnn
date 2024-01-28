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

def run(index, pc_name, pc, jpg_path):
    global pc_indices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc[0])
    point_cloud.colors = o3d.utility.Vector3dVector(pc[1])
    wrong = str(myWindow.entry3.get())
    truth = str(myWindow.entry4.get())
    name = str(index) + ' ' + pc_name + ' ' + wrong + '->' + truth
    vis = Visualization(name, point_cloud)
    p = ImgThread(jpg_path)
    p.start()
    time.sleep(1)
    vis.vis.run()
    picked = vis.vis.get_picked_points()
    pc_indices = [_.index for _ in picked]

def pub_pc(index):
    pc = pc_list[index]
    pc_path = os.path.join(data_dir, 'pcd_dir', pc_list[index])
    # pc_path = os.path.join(pcd_dir, pc)
    ann_path = pc_path.replace('pcd_dir', 'ann_dir')
    jpg_dir = os.path.dirname(pc_path.replace('pcd_dir', 'jpg_dir'))
    jpg_name = os.path.basename(pc_path).split('_')[0] + '.jpg'
    jpg_path = os.path.join(jpg_dir, jpg_name)
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

def change_entry(idx, text):
    myWindow.entry2.delete(0, END)
    myWindow.entry2.insert(END, str(idx))
    myWindow.entry1.delete(0, END)
    myWindow.entry1.insert(END, text)
    # t = img_list1[idx].strip().split('\t')[-1].split('->')
    # if len(t) > 1:
    #     myWindow.entry3.delete(0, END)
    #     # text = img_list[idx]
    #     myWindow.entry3.insert(END, t[0])
    #     myWindow.entry4.delete(0, END)
    #     # text = img_list[idx]
    #     myWindow.entry4.insert(END, t[1])

def up(event):
    idx = int(myWindow.entry2.get())
    if idx <= 0:
        return
    idx -= 1
    change_entry(idx, pc_list[idx])
    filter_data()

def down(event):
    idx = int(myWindow.entry2.get())
    if idx >= length - 1:
        return
    idx += 1
    change_entry(idx, pc_list[idx])
    filter_data()

def enter(event):
    # global img_list
    # path = myWindow.entry1.get()
    idx = int(myWindow.entry2.get())
    change_entry(idx, pc_list[idx])
    pub_pc(idx)

def filter_data():
    # path = myWindow.entry1.get()
    idx = int(myWindow.entry2.get())
    wrong = int(myWindow.entry3.get())
    pc_path = os.path.join(data_dir, 'pcd_dir', pc_list[idx])
    ann_path = pc_path.replace('pcd_dir', 'ann_dir')
    # pc_path = os.path.join(pcd_dir, pc)
    # ann_path = os.path.join(ann_dir, pc)
    jpg_dir = os.path.dirname(pc_path.replace('pcd_dir', 'jpg_dir'))
    jpg_name = os.path.basename(pc_path).split('_')[0] + '.jpg'
    jpg_path = os.path.join(jpg_dir, jpg_name)

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
    pc = pc_path.split('/')[-1]
    run(idx, pc, (points[label_valid, :], rgb[label_valid, :]), jpg_path)
    global pc_indices
    pc_indices = indices[pc_indices]

def change_label():
    idx = int(myWindow.entry2.get())
    if len(pc_indices) > 0:
        wrong = int(myWindow.entry3.get())
        truth = int(myWindow.entry4.get())
        # pc = pc_list[idx]
        ann_path = os.path.join(data_dir, 'pcd_dir', pc_list[idx])
        ann_path = ann_path.replace('pcd_dir', 'ann_dir')
        gt_seg_label = np.fromfile(ann_path, np.uint8).reshape(-1, 1)
        change_indices = [_ for _ in pc_indices if gt_seg_label[_] == wrong]
        gt_seg_label[change_indices] = truth
        gt_seg_label.tofile(ann_path)
        pub_pc(idx)

if __name__ == "__main__":
    data_path = sys.argv[1]
    data_dir = os.path.dirname(data_path)
    # pcd_dir = osp.join(data_dir, 'pcd_dir')
    # ann_dir = osp.join(data_dir, 'ann_dir')
    # jpg_dir = osp.join(data_dir, 'jpg_dir')
    try:
        img_list1 = open(data_path, 'r').readlines()
    except:
        print('未发现筛选数据，请先对数据进行筛选！')
        exit(-1)
    img_list1 = sorted(list(set(img_list1)))
    pc_list = [_.strip().split(' ')[0] for _ in img_list1]

    length = len(pc_list)
    myWindow = Window(length, filter_data, change_label, up, down, enter)
    if length == 0:
        exit(-1)
    
    last_idx = 0
    myWindow.entry2.insert(END, last_idx)
    # up(None)
        
    myWindow.run()
