#!/usr/bin/env python3
# -*- encoding utf-8 -*-

'''
@File: save_mnist_to_jpg.py
@Date: 2024-08-23
@Author: KRISNAT
@Version: 0.0.0
@Email: ****
@Copyright: (C)Copyright 2024, KRISNAT
@Desc: 
    1. 通过 torchvision.datasets.MNIST 下载、解压和读取 MNIST 数据集；
    2. 使用 PIL.Image.save 将 MNIST 数据集中的灰度图片以 JPEG 格式保存。
'''

import sys, os
sys.path.insert(0, os.getcwd())

from torchvision.datasets import MNIST
import PIL
from tqdm import tqdm

if __name__ == "__main__":
    root = 'mnist_jpg'
    if not os.path.exists(root):
        os.makedirs(root)

    train_dir = root + "/train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    test_dir = root + "/test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 从网络上下载或从本地加载MNIST数据集
    # 训练集60K、测试集10K
    # torchvision.datasets.MNIST接口下载的数据一组元组
    # 每个元组的结构是: (PIL.Image.Image image model=L size=28x28, 标签数字 int)
    training_dataset = MNIST(
        root='mnist',
        train=True,
        download=True,
    )
    test_dataset = MNIST(
        root='mnist',
        train=False,
        download=True,
    )

    # 保存训练集图片
    with tqdm(total=len(training_dataset), ncols=150) as pro_bar:
        for idx, (X, y) in enumerate(training_dataset):
            f = train_dir + "/" + "training_" + str(idx) + \
                "_" + str(training_dataset[idx][1] )+ ".jpg"  # 文件路径
            training_dataset[idx][0].save(f)
            pro_bar.update(n=1)

    # 保存测试集图片
    with tqdm(total=len(test_dataset), ncols=150) as pro_bar:
        for idx, (X, y) in enumerate(test_dataset):
            f = test_dir + "/" + "test_" + str(idx) + \
                "_" + str(test_dataset[idx][1] )+ ".jpg"  # 文件路径
            test_dataset[idx][0].save(f)
            pro_bar.update(n=1)
