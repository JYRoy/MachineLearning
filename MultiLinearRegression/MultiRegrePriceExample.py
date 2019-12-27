#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy
from _pydecimal import Decimal

import pymysql
from sklearn import datasets, linear_model
import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt

conn = pymysql.connect(
    host='47.99.185.152',
    user='root',
    password='A112323a.',
    database='bj_xiaozhu',
    charset='utf8'
)
# 游标
# cursor = conn.cursor()  # 执行完毕返回的结果集默认以元组显示
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 执行完毕返回的结果集默认以字典显示
cursor.execute('select district, house_type from rooms')
data_ana = cursor.fetchall()
cursor.execute('select floor_space from rooms')
data_floor_space = cursor.fetchall()
cursor.execute('select cash_pledge from rooms')
data_cash_pledge = cursor.fetchall()
cursor.execute('select room_number from rooms')
data_room_number = cursor.fetchall()
cursor.execute('select price from rooms')
price = cursor.fetchall()

encoder = ce.OneHotEncoder(cols=['district','house_type']).fit(data_ana)  # 处理哑变量
data_ana = encoder.transform(data_ana)
data_ana = np.array(data_ana)
print(data_ana)  # 哑变量

list_floor_space = []  # 存储房屋面积
list_cash_pledge = []  # 存储押金
list_room_number = []  # 存储房间类型
for item in data_floor_space:
    list_floor_space.append(item['floor_space'])
print(list_floor_space)
for item in data_cash_pledge:
    list_cash_pledge.append(item['cash_pledge'])
print(list_cash_pledge)
for item in data_room_number:
    list_room_number.append(item['room_number'])
print(list_room_number)

bins = [-1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300]
cuts_floor_space = pd.cut(list_floor_space, bins, right=True, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
print(cuts_floor_space)# 离散化之后的房屋面积

bins = [-1, 100, 200, 300, 400, 500, 1000, 2000]
cuts_cash_pledge = pd.cut(list_cash_pledge, bins, right=True, labels=[1, 2, 3, 4, 5, 6, 7])
print(cuts_cash_pledge)# 离散化之后的押金

X = []  # 数据集
Y = []  # 标签

for i in range(len(data_ana)):
    temp = []
    temp.extend(data_ana[i])
    temp.append(cuts_floor_space[i])
    temp.append(cuts_cash_pledge[i])
    temp.append(list_room_number[i])
    X.append(temp)
print(X)  # 数据集

for item in price:
    Y.append(int(item['price']))
print(Y)

# 8：2分割训练集和测集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
print("预测值：")
print(y_pred)
print("真实值：")
print(y_test)
# 折线图
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.show()




