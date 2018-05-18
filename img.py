#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
和图像操作有关函数
"""

import matplotlib.pyplot as plt
import numpy as np


def img2vector(filename):
    # 将图像转向量
    vector = np.zeros([1024], int)
    # 定义返回的向量，大小为1*1024
    lines = None
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 读取32*32数字文件
    for i in range(32):
        for j in range(32):
            vector[i * 32 + j] = lines[i][j]
    # 将信息存放在vector中
    return vector


def img2mat(filename):
    # 将图像转矩阵
    mat = np.zeros([32, 32], int)
    # 定义返回的矩阵，大小为32*32
    lines = None
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 读取32*32数字文件
    for i in range(32):
        for j in range(32):
            mat[i, j] = lines[i][j]
    # 将信息存放在mat中
    return mat


def show_img(mat):
    # 显示图像
    plt.imshow(mat)
    # plt.axis('off')
    plt.show()
