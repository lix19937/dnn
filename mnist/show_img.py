"""
通过gzip和numpy解析MNIST数据集的二进制文件, 并可视化训练集前10张图片和标签
"""

import os
import gzip
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(format="%(message)s", level=logging.DEBUG)  # 设置Python日志管理工具的消息格式和显示级别

plt.rcParams["font.sans-serif"] = "SimHei"  # 确保plt绘图正常显示中文
plt.rcParams["figure.figsize"] = [9, 10]  # 设置plt绘图尺寸

def parse_mnist(minst_file_addr: str = None) -> np.array:
    """解析MNIST二进制文件, 并返回解析结果
    输入参数:
        minst_file: MNIST数据集的文件地址. 类型: 字符串.

    返回值:
        解析后的numpy数组
    """
    if minst_file_addr is not None:
        minst_file_name = os.path.basename(minst_file_addr)  # 根据地址获取MNIST文件名字
        with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
            mnist_file_content = minst_file.read()
        if "label" in minst_file_name:  # 传入的为标签二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=8)  # MNIST标签文件的前8个字节为描述性内容，直接从第九个字节开始读取标签，并解析
        else:  # 传入的为图片二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=16)  # MNIST图片文件的前16个字节为描述性内容，直接从第九个字节开始读取标签，并解析
            data = data.reshape(-1, 28, 28)
    else:
        logging.warning(msg="请传入MNIST文件地址!")

    return data


if __name__ == "__main__":
    train_imgs = parse_mnist(minst_file_addr="train-images-idx3-ubyte.gz")  # 训练集图像
    train_labels = parse_mnist(minst_file_addr="train-labels-idx1-ubyte.gz")  # 训练集标签
    
    # 可视化
    fig, ax = plt.subplots(ncols=3, nrows=3)
    ax[0, 0].imshow(train_imgs[0], cmap=plt.cm.gray)
    ax[0, 0].set_title(f"标签为{train_labels[0]}")
    ax[0, 1].imshow(train_imgs[1], cmap=plt.cm.gray)
    ax[0, 1].set_title(f"标签为{train_labels[1]}")
    ax[0, 2].imshow(train_imgs[2], cmap=plt.cm.gray)
    ax[0, 2].set_title(f"标签为{train_labels[2]}")
    ax[1, 0].imshow(train_imgs[3], cmap=plt.cm.gray)
    ax[1, 0].set_title(f"标签为{train_labels[3]}")
    ax[1, 1].imshow(train_imgs[4], cmap=plt.cm.gray)
    ax[1, 1].set_title(f"标签为{train_labels[4]}")
    ax[1, 2].imshow(train_imgs[5], cmap=plt.cm.gray)
    ax[1, 2].set_title(f"标签为{train_labels[5]}")
    ax[2, 0].imshow(train_imgs[6], cmap=plt.cm.gray)
    ax[2, 0].set_title(f"标签为{train_labels[6]}")
    ax[2, 1].imshow(train_imgs[7], cmap=plt.cm.gray)
    ax[2, 1].set_title(f"标签为{train_labels[7]}")
    ax[2, 2].imshow(train_imgs[8], cmap=plt.cm.gray)
    ax[2, 2].set_title(f"标签为{train_labels[8]}")
    plt.show()  # 显示绘图

    print(plt.rcParams.keys())

