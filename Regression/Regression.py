# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

# function：加载数据
def loadData(filename):
    fr = open(filename)
    dataLines = fr.readlines()
    dataMat = []
    labelMat = []
    for line in dataLines:
        dataArr = line.strip().split('\t')
        dataMat.append([float(dataArr[0]), float(dataArr[1])])
        labelMat.append([float(dataArr[2])])
    return dataMat, labelMat


#####################################################  标准最小二乘回归  #################################################
# function：标准最小二乘计算参数
def standRegres(dataMat, labelMat):
    DataMatrix = mat(dataMat)
    LabelMatrix = mat(labelMat)
    xTx = DataMatrix.T * DataMatrix
    if linalg.det(xTx) == 0.0:  # 如果xTx的行列式为0，则表示矩阵不可逆
        print "The matrix is singular, can not inverse!"
        return
    else:
        w = xTx.I * (DataMatrix.T * LabelMatrix)
    return w


# function: 画示意图
def DrawFigure(dataMat, labelMat, w):
    DataMatrix = mat(dataMat)
    LabelMatrix = mat(labelMat)
    xcord = []
    for i in range(len(labelMat)):
        xcord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, labelMat, s = 30, c = 'blue', marker = 's')
    y_prect = DataMatrix[:, 0:2] * w
    print corrcoef(y_prect.T, LabelMatrix.T)  # 计算回归数据与正式数据之间的相关性
    ax.plot(DataMatrix[:, 1].flatten(), y_prect.T, 'r.')
    plt.show()


# function: 最小二乘执行函数
def LR():
    dataMat, labelMat = loadData("ex0.txt")
    w = standRegres(dataMat, labelMat)
    DrawFigure(dataMat, labelMat, w)


#####################################################  局部加权最小二乘回归  #############################################
# function：对某一个点进行加权最小二乘估计
def lwlr(testPoint, dataMat, labelMat, k = 1.0):
    DataMatrix = mat(dataMat)
    LabelMatrix = mat(labelMat)
    m = shape(DataMatrix)[0]
    weights = mat(eye((m)))
    for i in range(m):
        diff = testPoint - DataMatrix[i, :]
        weights[i,i] = exp(diff*diff.T / (-2.0*k**2))
    xTWx = DataMatrix.T * weights * DataMatrix
    if linalg.det(xTWx) == 0.0:
        print "The MArtix is singular!"
    else:
        y_predict = testPoint * (xTWx.I * (DataMatrix.T * weights * LabelMatrix))
    return y_predict


# function：对数据集进行加权最小二乘估计
def lwlrTest(testPointSet, dataMat, labelMat, k = 1.0):
    m = shape(testPointSet)[0]
    x = []
    y = []
    SortedIndex = mat(testPointSet).argsort(0)[:, 1]
    for index in SortedIndex.tolist():
        x.append(testPointSet[index[0]][1])
        y.extend(lwlr(testPointSet[index[0]], dataMat, labelMat, k).tolist()[0])
    return x, y

# function：绘制图形
def DrawFig(x, y, dataMat, labelMat):
    xcord = []
    for i in range(len(labelMat)):
        xcord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, mat(labelMat), s = 30, c = 'blue', marker = 's')
    ax.plot(x, y, c = 'red')
    plt.show()


# function: 局部加权最小二乘执行函数
def LocallyWeightedLR():
    dataMat, labelMat = loadData("ex0.txt")
    testPointSet = dataMat
    x, y = lwlrTest(testPointSet, dataMat, labelMat, k = 0.01)
    DrawFig(x, y, dataMat, labelMat)



LocallyWeightedLR()