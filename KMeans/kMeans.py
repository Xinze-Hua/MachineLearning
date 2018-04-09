# -*- coding:utf-8 -*-
import numpy as np
from numpy import *
import random


# 加载数据
def loadData(filename):
    dataSet = []
    fr = open(filename)
    dataLines = fr.readlines()
    for line in dataLines:
        lineArr = line.strip().split('\t')
        dataSet.append([float(el) for el in lineArr])
    return dataSet


# 计算距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 产生初始簇心(在数据集范围内产生)
def randCent(dataMat, k):
    n = shape(dataMat)[1]
    centriods = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataMat[:,j])
        rangeJ = float(max(dataMat[:,j]) - minJ)
        centriods[:, j] = minJ + rangeJ * np.random.rand(k ,1)
    return centriods


# k均值算法
def kMeans(dataMat, k, distMeas = distEclud, creatCent = randCent):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m, 2))) #  用来存储索引号和误差值
    centriods = creatCent(dataMat, k)  # 初始化质心
    clusterChanged = True
    while clusterChanged:  # 如果质心发生变化
        clusterChanged = False
        for i in range(m):  #  遍历数据集
            minDist = inf
            minIndex = -1
            for j in range(k):  # 寻找每个数据样本离得最近的质心点
                disJI = distMeas(dataMat[i, :], centriods[j, :])
                if disJI < minDist:
                    minDist = disJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2  # 更新样本点的信息
        # print centriods
        for cent in range(k):  # 更新质心
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # print ptsInClust
            centriods[cent, :] = mean(ptsInClust, axis=0)
    # print clusterAssment
    return centriods, clusterAssment


# 可视化表示聚类效果
def DrawFigure(DataSet, centriods, clusterAssment):
    import matplotlib.pyplot as plt
    colorIndex = ['red', 'blue', 'green', 'black', 'yellow']
    m = shape(DataSet)[0]  #  样本点数
    n = shape(centriods)[0]  # 类别数
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(n):
        ptsInClust = DataSet[nonzero(clusterAssment[:, 0].A == i)[0]]
        xCord = ptsInClust[:, 0].tolist()
        yCord = ptsInClust[:, 1].tolist()
        ax.scatter(xCord, yCord, color = colorIndex[i])
        ax.scatter(centriods[i, 0], centriods[i, 1], color=colorIndex[i], linewidth = 10)
    plt.show()


# 二分聚类
def bitKMeans(DataSet, k, distMeas = distEclud):
    m = shape(DataSet)[0]  # 样本数目
    clusterAssment = mat(zeros((m, 2)))
    centriod0 = mean(DataSet, axis = 0).tolist()[0]  #初始化簇心
    centList = [centriod0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(DataSet[j,:], mat(centriod0))
    while len(centList) < k: # 当前已分簇类小于k时继续进行分类
        lowerSSE = inf
        for i in range(len(centList)):
            ptsInClust = DataSet[nonzero(clusterAssment[:, 0].A == i)[0]]
            centriods, SplitclusterAssment = kMeans(ptsInClust, 2, distMeas)
            sseSplit = sum(SplitclusterAssment[:, 1]) # 划分部分的数据误差
            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 未划分部分数据的误差
            # print "sseSplit, and sseNoSplit is ", sseSplit, sseNoSplit
            if (sseSplit + sseNoSplit) < lowerSSE:
                bestCentToSplit = i  # 最佳划分簇
                bestNewCents = centriods
                bestClusterAssment = SplitclusterAssment.copy()
                lowerSSE = sseSplit + sseNoSplit  # 更新误差值
        bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 1)[0], 0] = len(centList)
        bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClusterAssment
    # print centList
    # print clusterAssment
    return  mat(centList), clusterAssment



DataSet = mat(loadData("testSet2.txt"))
# centriods, clusterAssment = kMeans(DataSet, 4)
# DrawFigure(DataSet, centriods, clusterAssment)
centriods, clusterAssment = bitKMeans(DataSet, 3)
DrawFigure(DataSet, centriods, clusterAssment)
# randCent(DataSet, 6)
# Distance = distEclud(DataSet[0], DataSet[0])
# print DataSet
# print Distance