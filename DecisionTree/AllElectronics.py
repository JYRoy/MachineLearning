#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree

#读取样本数据
allElectronicsData = open(r'AllElectronics.csv','r')
reader = csv.reader(allElectronicsData)
headers = next(reader)   #读取表头

print(headers)

featureList = []  #存放特征值
labelList = []   #存放目标标签

#将特征值和目标标签读取出来
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

#将数据转化成数字
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()
print("dummyX: " + str(dummyX))
#将数据转化成数字
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

#生成决策树模型
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

#预测
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictY = clf.predict([newRowX])
print("predictY: " + str(predictY))






