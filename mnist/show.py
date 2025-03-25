"""
通过gzip和numpy解析MNIST数据集的二进制文件
"""

import os
import gzip
import logging

import numpy as np

logging.basicConfig(format="%(message)s", level=logging.DEBUG)  # 设置Python日志管理工具的消息格式和显示级别


def parse_mnist(minst_file_addr: str = None, flatten: bool = False, one_hot: bool = False) -> np.array:
    """解析MNIST二进制文件, 并返回解析结果
    输入参数:
        minst_file: MNIST数据集的文件地址. 类型: 字符串.
        flatten: bool, 默认Fasle. 是否将图片展开, 即(n张, 28, 28)变成(n张, 784)
        one_hot: bool, 默认Fasle. 标签是否采用one hot形式.

    返回值:
        解析后的numpy数组
    """
    if minst_file_addr is not None:
        minst_file_name = os.path.basename(minst_file_addr)  # 根据地址获取MNIST文件名字
        with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
            mnist_file_content = minst_file.read()
        if "label" in minst_file_name:  # 传入的为标签二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=8)  # MNIST标签文件的前8个字节为描述性内容，直接从第九个字节开始读取标签，并解析
            if one_hot:
                data_zeros = np.zeros(shape=(data.size, 10))
                for idx, label in enumerate(data):
                    data_zeros[idx, label] = 1
                data = data_zeros
        else:  # 传入的为图片二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=16)  # MNIST图片文件的前16个字节为描述性内容，直接从第九个字节开始读取标签，并解析
            data = data.reshape(-1, 784) if flatten else data.reshape(-1, 28, 28)
    else:
        logging.warning(msg="请传入MNIST文件地址!")

    return data

if __name__ == "__main__":
    data = parse_mnist(minst_file_addr="./t10k-labels-idx1-ubyte.gz")  # t10k-images-idx1-ubyte.gz文件应该和本脚本在同一个目录下，否则应该修改地址
    print(len(data))  # 10000
    print(data[0:10])  # [7 2 1 0 4 1 4 9 5 9]

