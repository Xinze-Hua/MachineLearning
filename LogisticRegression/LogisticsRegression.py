# coding:utf-8
from numpy import *
import random

# 收集解析数据函数
def loadText():
    DataMat = [];   # 数据存储矩阵
    LabelMat = [];  # 标签存储矩阵
    fr = open("testSet.txt")   # 打开文件
    lines = fr.readlines()
    for line in lines:
        lineArr = line.strip().split()
        DataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        LabelMat.append(int(lineArr[2]))
    return DataMat, LabelMat


# sigmoid函数
def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))


# 梯度上升求参数函数
def gradAscend(DataMat, LabelMat):
    dataMatrix = mat(DataMat)  # 将数据转化numpy矩阵
    LabelMatrix = mat(LabelMat).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001  # 步长
    MaxCycles = 500  # 梯度迭代的次数
    Weight = ones((n,1))
    for k in range(MaxCycles):  # 根据逻辑斯蒂回归的学习策略进行推倒求出损失函数，然后得到梯度
        h = sigmoid(dataMatrix * Weight)
        error = LabelMatrix - h
        Weight = Weight + alpha * dataMatrix.transpose() * error
    return Weight


# 随机梯度下降法
def stocGradAscent0(DataMat, LabelMat):
    m, n = shape(DataMat)  # 获取数据集的维数
    weights = ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(sum(DataMat[i]*weights))
        error = LabelMat[i] - h
        weights = weights + [el*alpha*error for el in DataMat[i]]
    return weights


# 改进的随机梯度下降法
def stocGradAscent1(DataMat, LabelMat, numIter = 150):
    m, n = shape(DataMat)
    weights = ones(n) # 初始化权重系数
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(DataMat[randIndex] * weights))
            error = LabelMat[randIndex] - h
            weights = weights + [el * alpha * error for el in DataMat[randIndex]]
            del(dataIndex[randIndex])
    return weights


# 画出决策边界图
def plotBestFit():
    import matplotlib.pyplot as plt
    dataMat, LabelMat = loadText()
    # Weights = gradAscend(dataMat, LabelMat)
    # Weights = stocGradAscent0(dataMat, LabelMat)
    Weights = stocGradAscent1(dataMat, LabelMat)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = [];
    xcord2 = []; ycord2 = [];
    for i in range(n):
        if int(LabelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c = 'red',marker = 's')
    ax.scatter(xcord2,ycord2,s=30,c = 'green')
    x = arange(-3.0, 3.0, 0.1).reshape(1,60)
    y = (-Weights[0]-Weights[1]*x)/Weights[2]
    ax.plot(x, y, 'b*')
    plt.show()


def classifyVector(inX, weight):
    prob = sigmoid(sum(inX*weight))
    if prob > 0.5:return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainSet = []
    trainLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        trainLabel.append(float(currLine[21]))
    weight = stocGradAscent1(trainSet, trainLabel, 500)
    numTestCount = 0.0; errorCount = 0.0
    for line in frTest.readlines():
        numTestCount += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        result = classifyVector(lineArr,weight)
        if (int(result) != int(currLine[21])):
            errorCount += 1.0
    errorRate = errorCount/numTestCount
    print errorRate


colicTest()