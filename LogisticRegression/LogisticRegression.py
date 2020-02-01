#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

import numpy as np
import random

'''
LogisticRegression
'''

# 实现梯度下降算法，对应的公式实现
def gradientDescent(x, y, theta, alpha, m, numIterations):
    '''
    :param x: 下方创建的X
    :param y: 下方创建的Y
    :param theta: 公式中的参数值
    :param alpha: 学习率
    :param m: 实例的个数
    :param numIterations: 重复更新的次数
    :return: 
    '''
    xTrans = x.transpose()  # x的转置矩阵
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)  # 内积
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta



# 创建数据
def getData(numPoints, bias, variance):
    '''
    :param numPoints: 实例的数目
    :param bias: 偏向值
    :param variance: 方差
    :return: x, y
    '''
    x = np.zeros(shape=(numPoints, 2))  # shape 几行几列
    y = np.zeros(shape=(numPoints))

    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance

    return x, y

x, y = getData(100, 25, 10)
print("x:")
print(x)
print("Y:")
print(y)
m_x, n_x = np.shape(x)  # 看行列数
n_y = np.shape(y)
print("x shape:", m_x, n_x)
print("y shape:", n_y)

numIterations = 100000
alpha = 0.0005
theta = np.ones(n_x)
theta = gradientDescent(x, y, theta, alpha, m_x, numIterations)
print(theta)
