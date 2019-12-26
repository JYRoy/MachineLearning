#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

import numpy as np

def tanh(x):  # 双曲函数
    return  np.tanh(x)

def tanh_deriv(x):  # 双曲函数的导数
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):  # 逻辑函数
    return 1/(1 + np.exp(-x))


def logistic_derivative(x):  # 逻辑函数的导数
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    '''
    神经网络
    '''
    def __init__(self, layers, activation='tanh'):
        '''
        :param layers: 神经网络中的层数和每层神经元的数目，list
        :param activation: 使用的具体函数，默认用tanh
        '''
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []  # 初始化一个装权重的容器
        # 随机的产生权重
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)


    def fit(self, X, Y, learning_rate=0.2, epochs=10000):
        '''
        :param X: 训练集
        :param Y: 分类标记
        :param learning_rate: 学习率
        :param epochs: 设定的最大循环次数
        :return: 
        '''
        X = np.atleast_2d(X)  # 确认X的维度至少是两维的
        temp = np.ones([X.shape[0], X.shape[1]+1])  #  初始化一个矩阵，全是1， 行数：X的行数，列数：X的列数+1
        temp[:, 0:-1] = X
        X = temp  # 偏向的赋值
        Y = np.array(Y)  # 把Y转化成array类型

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # 随机抽取X中的一行
            a = [X[i]]
            #
            for l in range(len(self.weights)):
              a.append(self.activation(np.dot(a[l], self.weights[a[l]])))  # 更新下一层的相关节点

            error = Y[i] - a[-1]  # 计算误差
            deltas = [error*self.activation_deriv(a[-1])]  # 输出层的error

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)):  # 更新weight
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        '''
        :param x: 预测
        :return:  
        '''
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

