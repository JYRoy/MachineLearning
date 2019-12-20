#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

from sklearn import svm

x = [[2,0],[1,1], [2,3]]
y = [0, 0, 1]
clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)

print(clf)

# 查看支持向量是哪几个点
print(clf.support_vectors_)
#查看支持向量在x中的下标
print(clf.support_)
#查看一共几个点
print(clf.n_support_)