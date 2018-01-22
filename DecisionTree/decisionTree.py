# -*- coding:utf-8 -*-
import numpy
from math import log


FeatureLabels = ['Age', 'Work', 'House', 'Credit']
DataSet = [['0',0,0,0,'no'],['0',0,0,1,'no'],['0',1,0,1,'yes'],['0',1,1,0,'yes'],['0',0,0,0,'no'],['1',0,0,0,'no'],['1',0,0,1,'no'],
           ['1',1,1,1,'yes'],['1',0,1,2,'yes'],['1',0,1,2,'yes'],['2',0,1,2,'yes'],['2',0,1,1,'yes'],['2',1,0,1,'yes'],['2',1,0,2,'yes'],['2',0,0,0,'no']]


# FeatureLabels = ['No Surfacing', 'flippers', 'House', 'Credit']
# DataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]


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


# 按照特征进行数据划分
def splitDataSet(dataSet, featureAxis):
    '''This is a function that split the data set with the feature!'''
    SplitDataSet = []
    SplitDataProb = []
    dataArray = dataSet
    dataArray = numpy.array(dataArray) # 将数据集转换为numpy数组，方便后面的提取特征值
    splitValueVec = list(set(dataArray[:,featureAxis]))  # 提出对应特征的值组合
    # print splitValueVec
    for i in range(len(splitValueVec)):
        temp = []
        for el in dataSet:
            if(str(el[featureAxis])== splitValueVec[i]):
                temp.append(el)
        elProb = len(temp) / float(len(dataSet))
        SplitDataProb.append(elProb)
        SplitDataSet.append(temp)
    return SplitDataSet, SplitDataProb, splitValueVec


# 计算信息增益，选择最佳特征
def calcInformationGain(dataSet, featureVec):
    '''This is a function that compute the information gain of the given data set with each feature,
        and choose the optimal classify feature!'''
    entropy = calcShannonEnt(dataSet)  # 计算数据集的熵
    featureNum = len(featureVec) # 数据特征数目
    InformationGainVec = []
    for feature in featureVec:
        SplitDataSet, SplitDataProb, splitValueVec = splitDataSet(dataSet, int(feature))
        # print SplitDataSet
        # print SplitDataProb
        Ent_Exp = 0.0
        for i in range(len(SplitDataSet)):
            Ent = calcShannonEnt(SplitDataSet[i])
            # print Ent
            Ent_Exp += Ent * SplitDataProb[i]
        InformationGainVec.append(entropy - Ent_Exp)
    return featureVec[InformationGainVec.index(max(InformationGainVec))]

def MajorityCnt(classList):
    import operator
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


# 递归构建决策树
def creatDecisionTree(dataSet, featureVec):
    classList = [el[-1] for el in dataSet]
    if(classList.count(classList[0])==len(classList)):  #  如果数据集中的实例全部属于同一类，则停止递归
        # print 'hi'
        return classList[0]
    if(len(featureVec) == 0):
        return MajorityCnt(classList)
    bestFeatureIndex = calcInformationGain(dataSet, featureVec)
    DecisionTree = {FeatureLabels[bestFeatureIndex]:{}}
    featureVec.remove(bestFeatureIndex)
    print bestFeatureIndex
    print FeatureLabels[bestFeatureIndex]
    SplitDataSet, SplitDataProb, SplitValueVec = splitDataSet(dataSet, bestFeatureIndex)
    for value in SplitValueVec:
        DecisionTree[FeatureLabels[bestFeatureIndex]][value] = creatDecisionTree(SplitDataSet[SplitValueVec.index(value)], featureVec)
    return DecisionTree


print creatDecisionTree(DataSet,[0,1,2,3])