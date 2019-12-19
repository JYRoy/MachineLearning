#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy
import csv
import random
import math
import operator

# 加载数据集
def loadDataset(filename, split, trainingSet = [], testSet = []):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:  #将数据集随机划分
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# 计算点之间的距离，多维度的
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

# 获取k个邻居
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))   #获取到测试点到其他点的距离
    distances.sort(key=operator.itemgetter(1))    #对所有的距离进行排序
    neighbors = []
    for x in range(k):   #获取到距离最近的k个点
        neighbors.append(distances[x][0])
        return neighbors

# 得到这k个邻居的分类中最多的那一类
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]   #获取到票数最多的类别

#计算预测的准确率
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    #prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r'irisdata.txt', split, trainingSet, testSet)
    print('Trainset: ' + repr(len(trainingSet)))
    print('Testset: ' + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        # trainingsettrainingSet[x]
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    print('predictions: ' + repr(predictions))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()
