# -*- coding:utf-8 -*-
import numpy
from math import log


#####################################################数据集以及特征集#####################################################
FeatureLabels = ['Age', 'Work', 'House', 'Credit']
DataSet = [['0',0,0,0,'no'],['0',0,0,1,'no'],['0',1,0,1,'yes'],['0',1,1,0,'yes'],['0',0,0,0,'no'],['1',0,0,0,'no'],['1',0,0,1,'no'],
           ['1',1,1,1,'yes'],['1',0,1,2,'yes'],['1',0,1,2,'yes'],['2',0,1,2,'yes'],['2',0,1,1,'yes'],['2',1,0,1,'yes'],['2',1,0,2,'yes'],['2',0,0,0,'no']]

# FeatureLabels = ['No Surfacing', 'flippers', 'House', 'Credit']
# DataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]


#####################################################数据处理及特征提取###################################################
# 计算香农信息熵
def calcShannonEnt(dateSet):
    '''This is a function to compute the shannon Entropy of the dataSet!'''
    entropy = 0.0  # 数据集熵值初始化
    numEntries = len(dateSet)  # 获取实例个数
    labelCount = {}  # 用来记录各类别实例个数的字典
    for el in dateSet:  # 获得各类别实例个数
        currentLabel = el[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 1
        else:
            labelCount[currentLabel] += 1
    for key in labelCount:  # 计算熵值
        prob = float(labelCount[key]) / numEntries
        entropy -= prob * log(prob, 2)
    return entropy


# 按照特征进行数据划分
def splitDataSet(dataSet, featureAxis):
    '''This is a function that split the data set with the feature!'''
    SplitDataSet = []  # 存储划分后的数据集
    SplitDataProb = []  # 存储划分后的数据
    dataArray = dataSet
    dataArray = numpy.array(dataArray) # 将数据集转换为numpy数组，方便后面的提取特征值
    splitValueVec = list(set(dataArray[:,featureAxis]))  # 提出对应特征的值组合
    # print splitValueVec
    for i in range(len(splitValueVec)):  # 按照对应特征的各个值对数据集进行划分
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
    for feature in featureVec:  # 计算按照各个特征进行数据划分的条件熵值
        SplitDataSet, SplitDataProb, splitValueVec = splitDataSet(dataSet, int(feature))
        Ent_Exp = 0.0
        for i in range(len(SplitDataSet)):  # 计算条件熵值
            Ent = calcShannonEnt(SplitDataSet[i])
            Ent_Exp += Ent * SplitDataProb[i]
        InformationGainVec.append(entropy - Ent_Exp)
    return featureVec[InformationGainVec.index(max(InformationGainVec))]  # 返回最优分割特征


# 按照多数原则确定集合类别
def MajorityCnt(classList):
    '''This function is to decide which class the data set according to most principles '''
    import operator
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


#######################################################递归构造决策树#####################################################
# 递归构造决策树
def creatDecisionTree(dataSet, featureVec):
    classList = [el[-1] for el in dataSet]  # 数据的类别集合
    if(classList.count(classList[0])==len(classList)):  #  如果数据集中的实例全部属于同一类，则停止递归
        return classList[0]
    if(len(featureVec) == 0):  # 如果特征值用完了，停止递归
        return MajorityCnt(classList)
    bestFeatureIndex = calcInformationGain(dataSet, featureVec)  # 获取最优特征
    DecisionTree = {FeatureLabels[bestFeatureIndex]:{}}  # 构造树特征
    featureVec.remove(bestFeatureIndex)
    print bestFeatureIndex
    print FeatureLabels[bestFeatureIndex]
    SplitDataSet, SplitDataProb, SplitValueVec = splitDataSet(dataSet, bestFeatureIndex)
    for value in SplitValueVec:
        DecisionTree[FeatureLabels[bestFeatureIndex]][value] = creatDecisionTree(SplitDataSet[SplitValueVec.index(value)], featureVec)
    return DecisionTree



print creatDecisionTree(DataSet,[0,1,2,3])