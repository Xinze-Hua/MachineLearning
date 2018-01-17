# -*- coding:utf-8 -*-
from numpy import *
import numpy
import string
import matplotlib
import operator
import matplotlib.pyplot as plt

#  测试小程序
def creatDataSet():  # 生成数据
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2Martix(filename):  #  将文本记录转换为Numpy的解析程序
    fr = open(filename) # 打开文本
    arrayOfLines = fr.readlines()  # 逐行读取文本
    numberOfLines = len(arrayOfLines)  # 获取数据条数
    returnMat = zeros((numberOfLines,3))  # 建立一个空的数组，维数为数据个数*3
    classLabelVector = []  # 存储类别的数组
    index = 0
    for line in arrayOfLines:
        line = line.strip() # 删除一行文字中的指定字符，如果参数为空则默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        listFromLine = line.split('\t')  # 将每一行文字在'\'处进行分割，生成数组
        returnMat[index, :] = listFromLine[0:3]  # 将每一行数据存入先前申明的存储数组中
        if(listFromLine[-1] == 'largeDoses'):
            listFromLine[-1] = 3
        if(listFromLine[-1] == 'smallDoses'):
            listFromLine[-1] = 2
        if (listFromLine[-1] == 'didntLike'):
                listFromLine[-1] = 1
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def AnalysisWithFigue():   # 对数据的散点图进行分析
    datingDataMat, datingLabels = file2Martix('datingTestSet2.txt')  # 画图进行数据分析
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制散点图，第1和第2个参数是数据长度相同的数组,后面两个数组对应的分别是不同类别颜色和形状
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 5.0 * array(datingLabels),rray(datingLabels))
    plt.show()


def autoNorm(dataSet):  # 对数据进行归一化处理
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile([minVals], (m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):  # 对数据进行k近邻分类
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():  #  留一验证
    hoRatio = 0.10
    DatingDataMat, DatingLabels = file2Martix('datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(DatingDataMat)
    print normDataSet[0]
    num = normDataSet.shape[0]
    numTestVecs = int(num*hoRatio)  #选择数据集的数目
    errorCount = 0.0
    for i in range(numTestVecs):
        classfyResult = classify0(normDataSet[i], normDataSet[numTestVecs:num,:], DatingLabels[numTestVecs:num], 3)
        print "The classifier came back with: %d, the real answer is: %d" % (classfyResult, DatingLabels[i])
        if(classfyResult!=DatingLabels[i]):
            errorCount += 1.0
    print errorCount/float(numTestVecs)


def datingClassTest2():  # 交叉验证
    DatingDataMat, DatingLabels = file2Martix('datingTestSet.txt')
    normDataSet, ranges, minVals = autoNorm(DatingDataMat)
    DatingDataMat2, DatingLabels2 = file2Martix('datingTestSet2.txt')
    normDataSet2, ranges2, minVals2 = autoNorm(DatingDataMat2)
    num = normDataSet.shape[0]
    errorCount = 0.0
    for i in range(num):
        classfyResult = classify0(normDataSet[i], normDataSet2, DatingLabels2, 1)
        print "The classifier came back with: %d, the real answer is: %d" % (classfyResult, DatingLabels[i])
        if(classfyResult!=DatingLabels[i]):
            errorCount += 1.0
    print errorCount/float(num)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent of flier miles earned per year?"))
    iceCream = float(raw_input("litters of ice cream consumed per year?"))
    DatingDataMat, DatingLabels = file2Martix('datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(DatingDataMat)
    inArr = [ffMiles, percentTats, iceCream]
    temp = (inArr - minVals)/ranges
    classfyResult = classify0(temp, normDataSet, DatingLabels, 3)
    print resultList[classfyResult-1]


classifyPerson()