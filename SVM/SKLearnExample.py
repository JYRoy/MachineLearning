#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

'''
画图
'''

import numpy as np
import pylab as pl
from sklearn import svm

# 创建40个随机的点
np.random.seed(0) #取固定的一些随机数
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]] # 让两组点分别靠左和靠右，np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
Y = [0] * 20 + [1] * 20   # 赋给标记

# 实例化SVM分类器
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 求出超平面
# w0*x + w1*y + w3 = 0 可以转化成点斜式 y = -(w0/w1)*x + (w3/w1)
w = clf.coef_[0] # coef_为w1到wn
a = -w[0] / w[1]  #斜率
xx = np.linspace(-5, 5) # 产生 -5 到 5 中连续的值
yy = a * xx - (clf.intercept_[0]) / w[1]   # intercept_为w3

# 画出和支持向量点相切的两条直线
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print('w: ', w)
print('a: ', a)
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_", clf.coef_)

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

# 画出所有的点
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
# 生成图像
pl.axis('tight')
pl.show()