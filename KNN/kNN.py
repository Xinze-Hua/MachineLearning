# -*- coding:utf-8 -*-
from numpy import *
import numpy
import operator
def creatDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classfy0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 获取数据的个数，shape()函数用来获取数组维度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis = 0)
    print "zhongytukeyil"