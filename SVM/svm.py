# -*- coding:utf-8 -*-

from numpy import *
import random
import math

# function: 读取文本文件的数据
def LoadText(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    dataLines = fr.readlines()
    for line in dataLines:
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


################################################# 1.1 SVM - 简化版SMO算法 ###############################################


# function: 随机选取0-m中不等于i的数
def selectJInRandom(i, m):
    j = i
    if(j == i):
        j = int(random.uniform(0, m))
    return j


# function: alphaJ截断函数
def clipAlpha(alpha, L, H):
    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L
    return alpha


# function: 简化版SMO函数
def simpleSMO(dataMat, labelMat, C, toler, MaxIter):
    dataMatrix = mat(dataMat)  # 将特征数据转换为numpy数组
    labelMatrix = mat(labelMat).transpose()  # 将标签数据转换为numpy数组
    b = 0  # 初始化b的值
    iter = 0  # 初始化迭代次数
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))  # 初始化alpha值
    while(iter < MaxIter):  # 外层循环，表示循环了MaxIter次数都找不到需要优化的变量了，即表示优化结束
        alphaPairChanged = 0
        for i in range(m):  # 内部循环，遍历样本集，寻找需要优化的alpha，即不符合KKT条件的
            fxi = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fxi - float(labelMatrix[i])
            if ((Ei*labelMatrix[i] < -toler) and (alphas[i] < C))\
                    or ((Ei*labelMatrix[i] > toler) and (alphas[i] > 0)): # 样本点违反KKT条件
                j = selectJInRandom(i, m)
                fxj = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fxj - float(labelMatrix[j])
                alphasIold = alphas[i].copy()
                alphasJold = alphas[j].copy()
                if (labelMatrix[i] != labelMatrix[j]):
                    L = max(0, alphas[j]- alphas[i])
                    H = min(C, C + alphas[j]- alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print 'L==H'; continue
                eta = 2.0*(dataMatrix[i, :] * dataMatrix[j, :].T) - (dataMatrix[i, :] * dataMatrix[i, :].T)\
                      - (dataMatrix[j, :] * dataMatrix[j, :].T)
                if eta >= 0:
                    print 'eta>=0'; continue
                alphas[j] -= labelMatrix[j]*(Ei - Ej)/eta  # 更新alpha[j]的值
                alphas[j] = clipAlpha(alphas[j], L, H)
                if(abs(alphas[j] - alphasJold) < 0.00001):
                    print "J not moving enough!"; continue
                alphas[i] += labelMatrix[i]*labelMatrix[j]*(alphasJold - alphas[j])  # 更新alpha[i]的值
                b1 = b - Ei - labelMatrix[i]*(alphas[i] - alphasIold)*(dataMatrix[i, :] * dataMatrix[i, :].T)\
                     - labelMatrix[j]*(alphas[j] - alphasJold)*(dataMatrix[i, :] * dataMatrix[j, :].T)
                b2 = b - Ej - labelMatrix[i]*(alphas[i] - alphasIold)*(dataMatrix[i, :] * dataMatrix[j, :].T)\
                     - labelMatrix[j]*(alphas[j] - alphasJold)*(dataMatrix[j, :] * dataMatrix[j, :].T)
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif (alphas[j] > 0) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairChanged += 1
                print "iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairChanged)
        if(alphaPairChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    w = multiply(alphas, labelMatrix).T * dataMatrix
    return b, alphas, w


# function:画出分离示意图
def DrawFigure(dataMat, labelMat, b, w):
    import matplotlib.pyplot as plt
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    m, n = shape(dataMat)
    dataArr = array(dataMat)
    for i in range(m):
        if(int(labelMat[i]) == 1):  # 正实例
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:                       # 负实例
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(2, 6, 0.1)
    L = len(x)
    y = []
    for i in range(L):
        temp = -w[0, 0] / w[0, 1] * x[i] - b / w[0, 1]
        y.append(float(temp))
    ax.plot(x, y)
    plt.show()


# function：算法执行函数
def ExecuteFunction1():
    dataMat, labelMat = LoadText('testSet.txt')
    b, alphas, w = simpleSMO(dataMat, labelMat, 0.6, 0.001, 40)
    DrawFigure(dataMat, labelMat, b, w)

# ExecuteFunction1()


################################################# 1.2 SVM - Platt SMO算法 ##############################################

def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if(kTup[0] == 'lin'):  # 如果是线性模型
        K = X * A.T
    elif(kTup[0] == 'RBF'): # 如果是非线性模型，采用高斯核函数
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:print"error!"
    return K


#  定义一个对象用来管理数据
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn  # 特征数据集
        self.labelMat = classLabels  # 标签集
        self.C = C  # 调节参数
        self.tol = toler  # 松弛因子
        self.m = shape(dataMatIn)[0]  # 数据长度
        self.alphas = mat(zeros((self.m, 1)))  #  存储alpha数组
        self.b = 0  # 补偿
        self.eCache = mat(zeros((self.m, 2)))  # 误差缓存数组,第一个元素为标志位，第二个元素为E值
        self.K = mat(zeros((self.m, self.m))) #  用来存储每个样本经过核函数计算得到的函数值
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


# function: 计算Ei
def calcEk(os, k):
    fxk = float(multiply(os.alphas, os.labelMat).T * os.K[:, k]) + os.b
    Ek = fxk - float(os.labelMat[k])
    return Ek


# function：启发式寻找alphaJ
def selectJ(os, Ei, i): # 根据第一个alpha选择第二个alpha的值，这里采用启发式
    os.eCache[i] = [1, Ei]  # 更新误差列表
    validEacheList = nonzero(os.eCache[:,0].A)[0]  # nonzeros函数是用来提取数组中非零元素的索引，返回多维数组，分别记录非零元素的各个维度的索引
    maxDelta = 0.0
    maxK = -1;
    Ej = 0.0
    if(len(validEacheList)>0):
        for k in validEacheList:
            if k == i: continue  # 第二个alpha与第一个alpha不能相同
            Ek = calcEk(os, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDelta):
                maxK = k
                maxDelta = deltaE
                Ej = Ek
        return maxK, Ej
    else:                        # 第一次选择时，由于误差列表为0，所以进行随机选择j
        j = selectJInRandom(i, os.m);
        Ej = calcEk(os, j)
    return j, Ej


# function: 更新Ek
def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]


# functionL: 内部循环,选择第二个alpha值，并进行更新
def innerL(os, i):
    Ei = calcEk(os, i)
    if ((Ei*os.labelMat[i] < -os.tol) and (os.alphas[i] < os.C)) or ((Ei*os.labelMat[i] > os.tol) and (os.alphas[i] > 0)): # 第一个alpha满足KKT条件
        j, Ej = selectJ(os, Ei, i)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if(os.labelMat[i] != os.labelMat[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[i] + os.alphas[j])
        if L == H: print "L == H"; return 0
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j]
        if eta >= 0: print "eta > 0"; return 0
        os.alphas[j] -= os.labelMat[j]*(Ei - Ej)/eta
        os.alphas[j] = clipAlpha(os.alphas[j], L, H)
        updateEk(os, j)
        if(abs(os.alphas[j] - alphaJold) < 0.00001):
            print "J not moving enough!"; return 0
        os.alphas[i] += os.labelMat[i]*os.labelMat[j]*(alphaJold - os.alphas[j])
        updateEk(os, i)
        b1 = os.b - Ei - os.labelMat[i]*os.K[i, i]*(os.alphas[i] - alphaIold) \
             - os.labelMat[j]*os.K[i, j]*(os.alphas[j] - alphaJold)
        b2 = os.b - Ej - os.labelMat[i] * os.K[i, j] * (os.alphas[i] - alphaIold) \
             - os.labelMat[j] * os.K[j, j] * (os.alphas[j] - alphaJold)
        if (0 < os.alphas[i]) and (os.alphas[i] < os.C):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.alphas[j] < os.C):
            os.b = b2
        else:
            os.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


# functionL:外部循环,遍历样本，选择第一个alpha值
def smoP(dataMatIn, labelMatIn, C, toler, maxIter, kTup = ('lin', 0)):
    os = optStruct(mat(dataMatIn), mat(labelMatIn).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairChanged = 0
    while (iter < maxIter) and ((alphaPairChanged > 0) or (entireSet)):
        alphaPairChanged = 0
        if(entireSet):
            for i in range(os.m):
                alphaPairChanged += innerL(os, i)
                print "fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairChanged)
            iter += 1
        else:
            nonBoundIs = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0] # 找到所有大于0小于C的alpha的下标
            for i in nonBoundIs:
                alphaPairChanged += innerL(os, i)
                print "non-BoundSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairChanged)
            iter += 1
        if entireSet: entireSet = False
        elif(alphaPairChanged == 0): entireSet = True
        print "iterration Number: %d" % iter
    # w = multiply(os.alphas, os.labelMat).T * os.X
    return os.b, os.alphas


def DrawFigure2(dataMat, labelMat, b, Alphas):
    import matplotlib.pyplot as plt
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    SupportVx = []
    SupportVy = []
    SupportVIndex = nonzero(Alphas != 0)[0]
    m, n = shape(dataMat)
    dataArr = array(dataMat)
    for i in range(m):
        if (i in SupportVIndex): # 支持向量
            SupportVx.append(dataArr[i, 0])
            SupportVy.append(dataArr[i, 1])
            continue
        if (int(labelMat[i]) == 1):  # 正实例
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:  # 负实例
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    ax.scatter(SupportVx, SupportVy, s=30, c='blue', marker='*')
    plt.show()

def ExecuteFunction2():
    dataMat, labelMat = LoadText('testSetRBF.txt')
    errorCount = 0
    kTup = ('RBF', 1)
    b, Alphas = smoP(dataMat, labelMat, 200, 0.001, 10000, kTup)
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).transpose()
    SupportVIndex = nonzero(Alphas != 0)[0]
    SupportVector = dataMatrix[SupportVIndex]
    SupportVectorLabel = labelMatrix[SupportVIndex]
    print "there are %d Support Vector" % shape(SupportVector)[0]
    m, n = shape(dataMat)
    for i in range(m):
        kernel = kernelTrans(SupportVector, dataMatrix[i,:], kTup)
        predict = multiply(SupportVectorLabel, Alphas[SupportVIndex]).T * kernel + b
        if sign(predict) != labelMatrix[i]: errorCount += 1
    print "The Trainning error rate is %f" % (float(errorCount)/m)

    errorCount2 = 0
    DrawFigure2(dataMat, labelMat, b, Alphas)
    dataMat2, labelMat2 = LoadText('testSetRBF2.txt')
    dataMatrix2 = mat(dataMat2)
    labelMatrix2 = mat(labelMat2).transpose()
    for i in range(m):
        kernel = kernelTrans(SupportVector, dataMatrix2[i,:], kTup)
        predict = multiply(SupportVectorLabel, Alphas[SupportVIndex]).T * kernel + b
        if sign(predict) != labelMatrix2[i]: errorCount2 += 1
    print "The Ceshi error rate is %f" % (float(errorCount2)/m)

# ExecuteFunction2()


################################################# 1.3 SVM - 手写字体识别 ##############################################

def img2Vector(dirName, fileNameStr):
    dataMat = zeros((1, 1024))
    fr = open(dirName+fileNameStr)
    for i in range(32):
        dataLine = fr.readline()
        for j in range(32):
            dataMat[0, i*32 + j] = int(dataLine[j])
    return dataMat


def loadImage(dirName):
    from os import listdir
    trainningList = listdir(dirName)
    m = len(trainningList)
    trainningMat = zeros((m, 1024))
    hwLabels = []
    for i in range(m):
        fileNameStr = trainningList[i]
        fileStr =  fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if(classNumStr == 9): hwLabels.append(-1)
        else: hwLabels.append(1)
        trainningMat[i:,] = img2Vector(dirName, fileNameStr)
    return trainningMat, hwLabels


def HandWritingIdentify():
    dataMat, labelMat = loadImage('E:\Learning\MachineLearning\SVM\\trainingDigits\\')
    errorCount = 0
    kTup = ('RBF', 50)
    b, Alphas = smoP(dataMat, labelMat, 200, 0.001, 10000, kTup)
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).transpose()
    SupportVIndex = nonzero(Alphas != 0)[0]
    print SupportVIndex
    SupportVector = dataMatrix[SupportVIndex]
    SupportVectorLabel = labelMatrix[SupportVIndex]
    print "there are %d Support Vector" % shape(SupportVector)[0]
    m, n = shape(dataMat)
    for i in range(m):
        kernel = kernelTrans(SupportVector, dataMatrix[i,:], kTup)
        predict = multiply(SupportVectorLabel, Alphas[SupportVIndex]).T * kernel + b
        if sign(predict) != labelMatrix[i]: errorCount += 1
    print "The Trainning error rate is %f" % (float(errorCount)/m)

    errorCount2 = 0
    # DrawFigure2(dataMat, labelMat, b, Alphas)
    dataMat2, labelMat2 = loadImage('E:\Learning\MachineLearning\SVM\\testDigits\\')
    dataMatrix2 = mat(dataMat2)
    labelMatrix2 = mat(labelMat2).transpose()
    m2, n2 = shape(dataMat2)
    for i in range(m2):
        kernel = kernelTrans(SupportVector, dataMatrix2[i,:], kTup)
        predict = multiply(SupportVectorLabel, Alphas[SupportVIndex]).T * kernel + b
        if sign(predict) != labelMatrix2[i]: errorCount2 += 1
    print "The Ceshi error rate is %f" % (float(errorCount2)/m2)


HandWritingIdentify()