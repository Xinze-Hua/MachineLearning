# -*- coding: utf-8 -*-
from numpy import *

#####################################################  AdaBoost算法训练  ################################################
# function: 加载简单的样本数据
def loadSimpleData():
    dataMat = matrix([[1., 2.1],  # 数据集
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.],
                      ])
    labelMat = [1.0, 1.0, -1.0, -1.0, 1.0] # 标签集
    return dataMat, labelMat


# function: 类别判断函数
def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0
    return retArray


# function:构建决策树（即弱分类器）
def buildStump(dataMat, labelMat, D):  # 特征数据，标签数据，以及权值
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).T
    m, n = shape(dataMatrix)
    bestClasEst = mat(ones((m,1)))
    numSteps = 10  # 步数
    bestStump = {}  # 用来存储最佳决策树
    minError = inf # 记录最小误差值
    for i in range(n):  # 按特征进行遍历，即按列进行遍历
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        valueRange = rangeMax - rangeMin
        stepSize = valueRange / numSteps
        for j in range(-1, int(numSteps) + 1):  # 按照值进行遍历
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + stepSize * float(j))
                predictedVal = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVal == labelMatrix] = 0
                weightedError = D.T*errArr
                # print weightedError
                if weightedError < minError:
                    bestClasEst = predictedVal
                    minError = weightedError
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# function:构建AdaBoost决策树
def AdaBoostTrainDS(dataMat, labelMat, numIter):
    m, n = shape(dataMat)
    D = mat(ones((m, 1)) / m)
    weakClassArr = []
    aggClassEst = mat(zeros((m, 1)))
    for iter in range(numIter):
        bestStump, Error, ClasEst = buildStump(dataMat, labelMat, D)
        # bestStump['D'] = D
        # print 'D:', D.T
        # print 'ClassEst:', ClasEst
        alphas = float(0.5*log((1 - Error)/Error))
        bestStump['alphas'] = alphas
        weakClassArr.append(bestStump)
        expon =  multiply(-1*alphas*mat(labelMat).T, ClasEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()  # 更新权值
        aggClassEst += alphas*ClasEst
        aggerr = multiply(sign(aggClassEst) != mat(labelMat).T, ones((m, 1)))
        errRate = aggerr.sum() / m
        # print 'aggClassEst:', sign(aggClassEst)
        if(errRate == 0.0):
            break
    return weakClassArr, aggClassEst


# function:Adaboost分类函数
def AdaClassify(dataToClassify, classifierArr):
    dataMatrix = mat(dataToClassify)
    m = shape(dataMatrix)[0]
    aggClassEst = zeros((m, 1))
    for i in range(len(classifierArr)):
        classify_i = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst  += classify_i*classifierArr[i]['alphas']
    result_Class = sign(aggClassEst)
    # print "分类类别：", result_Class
    return result_Class, aggClassEst



# dataMat, labelMat = loadSimpleData()
# classifierArr = AdaBoostTrainDS(dataMat, labelMat, 9)
# AdaClassify([[5., 5.],[0., 0.]], classifierArr)


#######################################################  马疝病预测  ####################################################

# function: 加载样本数据
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    dataLines = fr.readlines()
    for line in dataLines:
        lineArr = line.strip().split('\t')
        temp = []
        for el in lineArr:
           temp.append(float(el))
        dataMat.append(temp[1:len(temp)-1])
        labelMat.append(temp[-1])
    return dataMat, labelMat


# function:ROC曲线的绘制
def plotROC(predStrengths, labelMat):
    import matplotlib.pyplot as plt
    cur = [1.0, 1.0]
    fig = plt.figure()
    numPosClas = sum(array(labelMat) == 1.0) # 病马的数目
    yStep = 1.0 / numPosClas
    xStep = 1.0 / (len(labelMat) - numPosClas)
    print predStrengths
    sortedIndicies = predStrengths.argsort(0)  # 按照从小到大进行排序，value越大，病马的可能性越大
    fig.clf()
    ax = fig.add_subplot(111)
    for index in sortedIndicies.tolist():
        print index
        if labelMat[index[0]] == 1:
            delx = 0
            dely = yStep
        else:
            delx = xStep
            dely = 0
        ax.plot([cur[0], cur[0] - delx], [cur[1], cur[1] - dely], c = 'b')
        cur = [cur[0] - delx, cur[1] - dely]
    ax.axis([0,1,0,1])
    plt.show()

dataMat, labelMat = loadDataSet('horseColicTraining2.txt')
classifierArr, aggClassEst1 = AdaBoostTrainDS(dataMat, labelMat, 50)
testDataSet, testLabelSet = loadDataSet('horseColicTest2.txt')
result_Class, aggClassEst2 = AdaClassify(testDataSet ,classifierArr)
ErrorSet = multiply(result_Class != mat(testLabelSet).T, ones((len(testLabelSet), 1)))
print "The error rate is ", ErrorSet.sum() / len(testLabelSet)
plotROC(aggClassEst1, labelMat)