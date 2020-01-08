#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

'''
简单线性回归手写源码
'''

import numpy as np

def fitSLR(x, y):
    '''
    根据传入的数据集计算回归方程的b0和b1
    :param x: 
    :param y: 
    :return: 
    '''
    n = len(x)
    dinominator = 0  # 分母, 计算b1用的
    numerator = 0  # 分子
    for i in range(0, n):
        numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        dinominator += (x[i] - np.mean(x))**2

    print("numerator: ", numerator)
    print("dinominator: ", dinominator)
    b1 = numerator/float(dinominator)
    b0 = np.mean(y)/float(np.mean(x))
    print("回归方程：", b0, '+ x *', b1)
    return b0, b1


def predict(x, b0, b1):
    '''
    根据回归方程预测
    :param x: 
    :param b0: 
    :param b1: 
    :return: 
    '''
    return b0 + x * b1

if __name__ == '__main__':
    x = [1, 3, 2, 1, 3]
    y = [14, 24, 18, 17, 27]
    b0, b1 = fitSLR(x, y)
    print(predict(4, b0, b1))