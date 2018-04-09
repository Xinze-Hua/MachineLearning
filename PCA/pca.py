# -*- coding:utf-8 -*-

from numpy import *


##################################################### 1.1 PCA算法 #######################################################

# 导入数据，以矩阵的形式返回数据集
def loadData(filename):
    fr = open(filename)
    stringArr = [el.strip().split() for el in fr.readlines()]
    dataArr = [map(float, el) for el in stringArr]
    return mat(dataArr)


# pca降维处理
def pca(DataMat, topNfeat = 99999):
    meanVal = mean(DataMat, axis = 0)  # axis = 0表示按列球平均值，axis = 1表示按行求平均值
    meanRemoved = DataMat - meanVal
    covMat = cov(meanRemoved, rowvar = 0)
    eigVal, eigVec = linalg.eig(covMat)
    eigValInd =  argsort(eigVal)
    eigValInd = eigValInd[-1:-(topNfeat+1): -1]  # 反向取topNfeat个数
    redEigVec = eigVec[:, eigValInd]
    lowDDataMat =  meanRemoved*redEigVec
    reconDataMat = lowDDataMat*redEigVec.T + meanVal
    return lowDDataMat, reconDataMat


# 绘制数据散点图
def DrawFigure(DataMat, reconDataMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(DataMat[:, 0].A, DataMat[:, 1].A, color = 'r', marker = '^')
    ax.scatter(reconDataMat[:, 0].A, reconDataMat[:, 1].A, color='g', marker='o')
    plt.show()


############################################### 1.2 对半导体制造数据进行降维 ##############################################

# 将空数据进行替换
def replaceNanWithMean():
    DataMat = loadData("secom.data")
    numFeat = shape(DataMat)[1]
    for i in range(numFeat):
        meanVal = mean(DataMat[nonzero(~isnan(DataMat[:,i].A))[0], i])
        DataMat[nonzero(isnan(DataMat[:, i].A))[0], i] = meanVal
    return DataMat

DataMat = replaceNanWithMean()
lowDDataMat, reconDataMat = pca(DataMat, 3)
print lowDDataMat
# DataMat = loadData("testSet.txt")
# lowDDataMat, reconDataMat = pca(DataMat, 1)
# DrawFigure(DataMat, reconDataMat)
