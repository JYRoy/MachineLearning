#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

from time import time
import logging  # 打印process
import matplotlib.pyplot as plt # 为了把人脸画出来

from sklearn.model_selection  import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

print(__doc__)

# 打印程序process信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# 下载名人库的人脸数据集
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 获取数据集的实例数量，
n_samples, h, w = lfw_people.images.shape

# 获取特征向量矩阵
X = lfw_people.data
# 获取特征向量的维度,即列数
n_features = X.shape[1]

# 获取class label 目标怕分类标记
Y = lfw_people.target
# 获取到所有的名字
target_names = lfw_people.target_names
# 获取总数，即行数
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# 把数据集拆分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

'''
利用PCA降维，把特征向量的维度降下来，以提高做预测的真确性
'''
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
# PCA算法实例化
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
# 在人脸的一些照片上提取一些特征值
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
# 降维
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

'''
分类器分类，SVM实现
'''
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],     # 对错误参数进行惩罚的权重
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }   # 多少的特征点将会被使用
# 找到表现最好的一组函数的分类器
clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

'''
预测
'''
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
# 把预测的标签和真实的标签做比较
print(classification_report(y_test, y_pred, target_names=target_names))
# 建立一个对比的矩阵
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


'''
使用matplotlib对预测进行定性评估
'''
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


'''
将预测结果绘制在测试集的一部分上
'''
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# 绘制最重要特征脸的图库
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
