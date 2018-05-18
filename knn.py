#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
k-NearestNeighbor
k近临算法的python实现
"""

import numpy as np


def classify(X, DATASET, LABELS, K):
    # 大写参数表示常量，不可改变
    distances = np.sqrt(np.sum(np.square(DATASET - X), axis=1))
    # 计算距离矩阵
    len_dis = len(distances)
    # 得到distances数目
    labels = []
    # 存储标签
    for i in range(0, K):
        min_value = distances[i]
        min_value_idx = i
        for j in range(i + 1, len_dis):
            if distances[j] < min_value:
                min_value = distances[j]
                min_value_idx = j
        distances[i], distances[min_value_idx] = distances[min_value_idx], distances[i]
        labels.append(LABELS[min_value_idx])
    # 选择排序挑选出前k个最值
    # 用labels存储前k个最小距离的标签
    C = labels[0]
    max_count = 0
    for label in labels:
        count = labels.count(label)
        if count > max_count:
            max_count = count
            C = label
    # 求前k个label中，重复次数最多的label，并返回
    return C
