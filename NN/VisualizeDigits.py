#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

from sklearn.datasets import load_digits
import pylab as pl
'''
数字数据集
'''
if __name__ == '__main__':
    digits = load_digits()
    print(digits.data.shape)

    pl.gray()
    pl.matshow(digits.images[0])
    pl.show()
