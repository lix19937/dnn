#! /usr/bin/python
# -*- coding: utf-8 -*-

#1.导包 
import sys
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image as ImgMsg
import os
import os.path as osp
import numpy as np
import cv2
import cv_bridge
from pt_vis import Window
try:
    from Tkinter import *
except:
    from tkinter import *

color_dict = [[255, 0, 0], [255, 127, 80], [218, 97, 17],
           [255, 255, 0], [0, 200, 200], [128, 128, 0], [200, 200, 200],
           [107, 142, 35], [0, 255, 127], [152, 251, 152], [100, 60, 255], [142, 0, 252],
           [0, 0, 255], [60, 179, 113], [255, 215, 0], [0, 60, 100],
           [0, 80, 100], [72, 51, 159], [119, 11, 32], [128, 64, 128]]

def pub_pc(i):
    img = img_list[i]
    pc_path = os.path.join(img_dir, img)
    ann_path = os.path.join(ann_dir, img)
    points = np.fromfile(pc_path, np.float32).reshape(-1, 3)
    gt_seg_label = np.fromfile(ann_path, np.uint8).reshape(-1, 1)
    rgb = np.ones((points.shape[0], 3), np.float32)
    for t in range(points.shape[0]):
        rgb[t, :] = np.asarray(color_dict[gt_seg_label[t, 0]], np.float32) / 255
    points = np.concatenate((points, rgb, gt_seg_label), axis=1)

    msg.row_step = msg.point_step * points.shape[0]
    msg.width = points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points, np.float32).tostring()

    pub.publish(msg) #发布信息到主题
    jpg_path = os.path.join(jpg_dir, img.split('_')[0] + '.jpg')
    if not osp.exists(jpg_path):
        return
    jpg = cv2.imread(jpg_path)
    img_msg = bridge.cv2_to_imgmsg(jpg, encoding='bgr8')
    pub_img.publish(img_msg)
    # print('file name: ' + img + ' ' + str(i) + '/' + str(length))

def up(event):
    idx = int(myWindow.entry2.get())
    if idx <= 0:
        return
    idx -= 1
    while img_list[idx] in filter_name:
        idx -= 1
        if idx < 0:
            return
    myWindow.entry2.delete(0, END)
    myWindow.entry2.insert(END, str(idx))
    myWindow.entry1.delete(0, END)
    text = img_list[idx]
    myWindow.entry1.insert(END, text)
    pub_pc(idx)

def down(event):
    idx = int(myWindow.entry2.get())
    if idx >= length - 1:
        return
    idx += 1
    while img_list[idx] in filter_name:
        idx += 1
        if idx > length - 1:
            return
    myWindow.entry2.delete(0, END)
    myWindow.entry2.insert(END, str(idx))
    myWindow.entry1.delete(0, END)
    text = img_list[idx]
    myWindow.entry1.insert(END, text)
    pub_pc(idx)

def enter(event):
    idx = int(myWindow.entry2.get())
    myWindow.entry1.delete(0, END)
    text = img_list[idx]
    myWindow.entry1.insert(END, text)
    pub_pc(idx)

def filter_data():
    global filter_name
    file_path = str(myWindow.entry1.get())
    file_name = file_path + ' ' + str(myWindow.entry2.get()) + '\t' + str(myWindow.entry3.get()) + '->' + str(myWindow.entry4.get())
    filter_name[file_path] = file_name + '\n'
    filter_file.write(file_name + '\n')
    filter_file.flush()
    down(None)
    print('write ' + file_name + ' to filter.txt')

def filter_severe_data():
    global filter_name
    global filter_severe_name
    file_path = str(myWindow.entry1.get())
    file_name = file_path + ' ' + str(myWindow.entry2.get()) + '\t' + str(myWindow.entry3.get()) + '->' + str(myWindow.entry4.get())
    filter_name[file_path] = file_name + '\n'
    filter_severe_name[file_path] = file_name + '\n'
    filter_file.write(file_name + '\n')
    filter_file.flush()
    filter_severe_file.write(file_name + '\n')
    filter_severe_file.flush()
    down(None)
    print('write ' + file_name + ' to filter.txt and filter_severe.txt')

if __name__ == '__main__':
    #2.初始化 ROS 节点:命名(唯一)，节点为talker_p
    rospy.init_node('talker_p')
    #3.实例化 发布者 对象(发布话题-chatter，std_msgs.msg.String类型,队列条目个数)
    pub =rospy.Publisher('pc', PointCloud2, queue_size=10)
    pub_img =rospy.Publisher('img', ImgMsg, queue_size=10)
    data_dir = sys.argv[1]
    img_dir = os.path.join(data_dir, 'pcd_dir')
    ann_dir = os.path.join(data_dir, 'ann_dir')
    jpg_dir = os.path.join(data_dir, 'jpg_dir')
    img_list = sorted(os.listdir(img_dir))

    filter_path = os.path.join(data_dir, 'filter.txt')
    filter_name = dict()
    if os.path.exists(filter_path):
        filtered = open(filter_path, 'r').readlines()
        filter_name = [_.split(' ')[0].strip() for _ in filtered]
        filter_name = [_.split('/')[-1] for _ in filter_name]
        filter_name = dict(zip(filter_name, filtered))
    filter_severe_path = os.path.join(data_dir, 'filter_severe.txt')
    filter_severe_name = dict()
    if os.path.exists(filter_severe_path):
        filtered = open(filter_severe_path, 'r').readlines()
        filter_severe_name = [_.split(' ')[0].strip() for _ in filtered]
        filter_severe_name = [_.split('/')[-1] for _ in filter_severe_name]
        filter_severe_name = dict(zip(filter_severe_name, filtered))

    bridge = cv_bridge.CvBridge()
    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = "base_link"
    
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1),
        PointField('label', 24, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 28
    msg.height = 1
    length = len(img_list)
    # print('all data count: {}'.format(length))
    
    myWindow = Window(length, filter_data, filter_severe_data, up, down, enter)
    if len(filter_name) == 0:
        last_idx = 1
        myWindow.entry2.insert(END, last_idx)
        up(None)
    else:
        last_idx = sorted(filter_name.values())[-1].split(' ')[1].split('\t')[0]
        myWindow.entry2.insert(END, last_idx)
        down(None)
    filter_file = open(filter_path, 'a')
    filter_severe_file = open(filter_severe_path, 'a')
    myWindow.run()   #
    filter_file.close()
    filter_severe_file.close()

    filter_file = open(filter_path, 'w')
    for key in sorted(filter_name.keys()):
        filter_file.write(filter_name[key])
    filter_file.close()
    filter_severe_file = open(filter_severe_path, 'w')
    for key in sorted(filter_severe_name.keys()):
        filter_severe_file.write(filter_severe_name[key])
    filter_severe_file.close()

