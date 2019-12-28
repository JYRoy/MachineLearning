#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

from NeuralNetworkImplement import NeuralNetwork
import numpy as np
'''
利用简单非线性关系的数据集测试Implement
'''
if __name__ == '__main__':
    nn = NeuralNetwork([2, 2, 1], 'tanh')  # 输入层2个神经元， 隐藏层2个神经元， 输出层1个神经单元
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    nn.fit(X, Y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(i, nn.predict(i))

