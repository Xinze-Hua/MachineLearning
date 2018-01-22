# -*- coding:utf-8 -*-
import numpy
from math import log


# 计算香农信息熵
def calcShannonEnt(dateSet):
    '''This is a function to compute the shannon Entropy of the dataSet!'''
    numEntries = len(dateSet)
    labelCount = {}
    for el in dateSet:
        currentLabel = el[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 1
        else:
            labelCount[currentLabel] += 1
    entropy = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries
        entropy -= prob * log(prob, 2)
    return entropy

# DataSet = [[1,1,'yes'],[1,0,'yes'],[1,1,'no'],[1,1,'no'],[1,1,'no']]


# 按照特征进行数据划分
def splitDataSet(dataSet, featureAxis):
    '''This is a function that split the data set with the feature!'''
    SplitDataSet = []
    SplitDataProb = []
    dataArray = dataSet
    dataArray = numpy.array(dataArray) # 将数据集转换为numpy数组，方便后面的提取特征值
    splitValueVec = list(set(dataArray[:,featureAxis]))  # 提出对应特征的值组合
    for i in range(len(splitValueVec)):
        temp = []
        for el in dataSet:
            if(el[featureAxis]==int(splitValueVec[i])):
                temp.append(el)
        elProb = len(temp) / float(len(dataSet))
        SplitDataProb.append(elProb)
        SplitDataSet.append(temp)
    return SplitDataSet, SplitDataProb


# 计算信息增益，选择最佳特征
def calcInformationGain(dataSet, featureVec):
    '''This is a function that compute the information gain of the given data set with each feature,
        and choose the optimal classify feature!'''
    entropy = calcShannonEnt(dataSet)  # 计算数据集的熵
    featureNum = len(featureVec) # 数据特征数目
    InformationGainVec = []
    for feature in featureVec:
        SplitDataSet, SplitDataProb = splitDataSet(dataSet, int(feature))
        print SplitDataSet
        print SplitDataProb
        Ent_Exp = 0.0
        for i in range(len(SplitDataSet)):
            Ent = calcShannonEnt(SplitDataSet[i])
            print Ent
            Ent_Exp += Ent * SplitDataProb[i]
        InformationGainVec.append(entropy -Ent_Exp)
    print InformationGainVec





DataSet = [[0,1,'yes'],[1,0,'yes'],[1,1,'no'],[1,1,'no'],[1,1,'no'],[1,2,'yes'],[1,0,'yes'],[3,2,'no'],[1,1,'no'],[1,1,'no']]
# print calcShannonEnt(DataSet)
# print splitDataSet(DataSet, 1, [0,1,2])
calcInformationGain(DataSet,[0,1])
