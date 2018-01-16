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
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 计算各个点与目标点在每个维度上的差值
    sqDiffMat = diffMat**2  # 计算差值的平方
    sqDistance = sqDiffMat.sum(axis = 1)   # 将每个点在各个维度上与目标点的差值平方相加
    distance = sqDistance**0.5  # 进行开方处理，11,12,13,14,15是用来计算训练点与目标点的欧式距离
    sortedDistIndicies = distance.argsort() # 按距离从小大进行排序，并记录各个点排序前的索引值
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1  # 相同的类别进行累加
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
Group, Labels = creatDataSet()
print classfy0([1.3, 1.4], Group, Labels, 3)

