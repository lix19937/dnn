#! /usr/bin/python
# -*- coding: utf-8 -*-

#1.导包 
import sys
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image as ImgMsg
import cv2
import cv_bridge
import os
import os.path as osp
import numpy as np
try:
    from Tkinter import *
except:
    from tkinter import *

# from view_lidar import color_dict

color_dict = [[255, 0, 0], [255, 127, 80], [218, 97, 17],
           [255, 255, 0], [0, 200, 200], [128, 128, 0], [200, 200, 200],
           [107, 142, 35], [0, 255, 127], [152, 251, 152], [100, 60, 255], [142, 0, 252],
           [0, 0, 255], [60, 179, 113], [255, 215, 0], [0, 60, 100],
           [0, 80, 100], [72, 51, 159], [119, 11, 32], [128, 64, 128]]

def pub_pc(i):
    global img_list
    img = os.path.join(data_dir, 'pcd_dir', img_list[i])
    pc_path = img
    ann_path = img.replace('pcd_dir', 'ann_dir')
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

    jpg_dir = os.path.dirname(img.replace('pcd_dir', 'jpg_dir'))
    jpg_name = os.path.basename(img).split('_')[0] + '.jpg'
    jpg_path = os.path.join(jpg_dir, jpg_name)
    if os.path.exists(jpg_path):
        jpg = cv2.imread(jpg_path)
        img_msg = bridge.cv2_to_imgmsg(jpg, encoding='bgr8')
        pub_img.publish(img_msg)

def change_entry(idx, text):
    entry2.delete(0, END)
    entry2.insert(END, str(idx))
    entry1.delete(0, END)
    entry1.insert(END, text)
    t = img_list1[idx].strip().split('\t')[-1].split('->')
    if len(t) > 1:
        entry3.delete(0, END)
        # text = img_list[idx]
        entry3.insert(END, t[0])
        entry4.delete(0, END)
        # text = img_list[idx]
        entry4.insert(END, t[1])

def up(event):
    global img_list
    idx = int(entry2.get())
    if idx <= 0:
        return
    idx -= 1
    change_entry(idx, img_list[idx])
    pub_pc(idx)

def down(event):
    global img_list
    idx = int(entry2.get())
    if idx >= length - 1:
        return
    idx += 1
    change_entry(idx, img_list[idx])
    pub_pc(idx)

def right(event):
    global img_list
    idx = int(entry2.get())
    if idx >= length - 10:
        return
    idx += 10
    change_entry(idx, img_list[idx])
    pub_pc(idx)

def left(event):
    global img_list
    idx = int(entry2.get())
    if idx < 10:
        return
    idx -= 10
    change_entry(idx, img_list[idx])
    pub_pc(idx)

def enter(event):
    global img_list
    path = entry1.get()
    print(path)
    idx = img_list.index(path)
    print(idx)
    change_entry(idx, path)
    pub_pc(idx)

def filter_data():
    global img_list
    global img_list1
    global length
    idx = int(entry2.get())
    if idx < 0 or idx > length - 1:
        print('idx error!')
        return
    file_name = img_list[idx]
    del img_list1[idx]
    img_list = [_.strip().split(' ')[0] for _ in img_list1]
    
    entry2.delete(0, END)
    entry2.insert(END, str(idx-1))
    length = len(img_list1)
    frame['text'] = "all data count:{}".format(length)
    filter_file = open(data_path, 'w')
    [filter_file.write(_) for _ in img_list1]
    filter_file.close()
    down(None)

def select_data():
    file_name = str(entry1.get())
    write_path = osp.join(osp.dirname(data_path), 'select.txt')
    write_file = open(write_path, 'a')
    write_file.write(file_name + '\n')
    write_file.close()
    down(None)

if __name__ == '__main__':
    #2.初始化 ROS 节点:命名(唯一)，节点为talker_p
    rospy.init_node('talker_p')
    #3.实例化 发布者 对象(发布话题-chatter，std_msgs.msg.String类型,队列条目个数)
    pub =rospy.Publisher('pc', PointCloud2, queue_size=10)
    pub_img =rospy.Publisher('img', ImgMsg, queue_size=10)
    data_path = sys.argv[1]
    start_frame = 0
    if len(sys.argv) > 2:
        start_frame = int(sys.argv[2])
    global img_list
    global img_list1
    data_dir = os.path.dirname(data_path)
    try:
        img_list1 = open(data_path, 'r').readlines()
    except:
        print('未发现筛选数据，请先对数据进行筛选！')
        exit(-1)
    img_list1 = sorted(list(set(img_list1)))
    img_list = [_.strip().split(' ')[0] for _ in img_list1]
    if start_frame < len(img_list):
        img_list = img_list[start_frame:]

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
    if length == 0:
        print('未发现筛选数据，请先对数据进行筛选！')
        exit(-1)
    # print('all data count: {}'.format(length))
    
    myWindow = Tk()
    myWindow.title("seg select")    # #窗口标题
    myWindow.resizable(False, False)
    myWindow.geometry("+500+140")   # #窗口位置500后面是字母x
    myWindow.attributes("-topmost",1)
    # myWindow.geometry("600x500+20+20")   # #窗口位置500后面是字母x
    #标签控件布局
    frame = LabelFrame(myWindow, text="all data count:{}".format(length), font=('Arial 12 bold'))
    frame.grid(row=0,sticky=W, padx=5, pady=5)
    xlabel1 = Label(frame, text="file name:", font=('Arial 12 bold'), width=20, height=5)
    xlabel1.grid(row=0)
    Label(frame, text="input", font=('Arial 12 bold'), width=20, height=5).grid(row=1)
    #标签控件布局
    frame = LabelFrame(myWindow, text="all data count:{}".format(length), font=('Arial 12 bold'))
    frame.grid(row=0,sticky=W, padx=5, pady=5)
    xlabel1 = Label(frame, text="file name:", font=('Arial 12 bold'), width=10, height=2)
    xlabel1.grid(row=0)
    Label(frame, text="input:", font=('Arial 12 bold'), width=10, height=2).grid(row=1)
    #Entry控件布局
    entry1=Entry(frame, font=('Arial 12'), width=30)
    def validate(text):
        if isinstance(text, str) and text.isdigit():
            return True
        else:
            return False
    v_cmd = (frame.register(validate), '%S')
    entry2=Entry(frame, font=('Arial 12'), width=30, validate='key', validatecommand=v_cmd)
    entry2.insert(END, str(0))
    entry1.grid(row=0, column=1, columnspan=2)
    entry2.grid(row=1, column=1, columnspan=2)

    Label(frame, text="wrong:", font=('Arial 12 bold'), width=10, height=2).grid(row=2)
    Label(frame, text="truth:", font=('Arial 12 bold'), width=10, height=2).grid(row=3)
    entry3=Entry(frame, font=('Arial 12'), width=30, validate='key', validatecommand=v_cmd)
    entry3.insert(END, '0')
    entry4=Entry(frame, font=('Arial 12'), width=30, validate='key', validatecommand=v_cmd)
    entry4.insert(END, '12')
    entry3.grid(row=2, column=1, columnspan=2)
    entry4.grid(row=3, column=1, columnspan=2)

    Button(frame, text='Quit', font=('Arial 12 bold'), width=10, height=2, command=myWindow.quit).grid(row=4, column=0,sticky=W, padx=5, pady=5)
    Button(frame, text='Filter', font=('Arial 12 bold'), width=10, height=2, command=filter_data).grid(row=4, column=1, sticky=W, padx=5, pady=5)
    Button(frame, text='Select', font=('Arial 12 bold'), width=10, height=2, command=select_data).grid(row=4, column=2, sticky=W, padx=5, pady=5)

    myWindow.bind("w", up)
    myWindow.bind("s", down)
    myWindow.bind("d", right)
    myWindow.bind("a", left)
    myWindow.bind("<Return>", enter)
    
    entry2.insert(END, str(1))
    up(None)
    myWindow.mainloop()   # #窗口持久化

