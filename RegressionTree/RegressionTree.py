# coding:utf-8
# 第一步：整理数据，将样本数据以矩阵的形式进行存储
# 第二步：按照特征和特征值进行数据分割，
# 第三步：计算基尼指数，选择最优切割特征以及所对应的特征值，
# 第四步：递归构造回归树

import numpy as np
import operator

FEATURE = ['X']


####################################################### 数据处理部分 ####################################################

#  样本文本导入与数据切分
def LoadDataSet(filename):
    '''This function is to load file and split the text!'''
    fr = open(filename)
    DataList = fr.readlines()
    DataMat = []  # 样本特征数据
    for line in DataList:
        line = line.strip().split('\t')
        line = [float(item) for item in line]
        DataMat.append(line)
    return DataMat


#  按照指定特征以及对应的特征值进行文本划分
def SplitDataSet(DataSet, feature, value):
    '''This function  is to split data set whih the given feature and value!'''
    featureIndex = FEATURE.index(feature)  # 获取当前的特征所对应FEATURE的索引值
    LeftDataSet = []
    RightDataSet = []
    for Data in DataSet:
        if Data[featureIndex] > value:  # 对应特征的值大于value则放在LeftDataSet中
            LeftDataSet.append(Data)
        else:                           # 对应特征的值小于等于value则放在RightDataSet中
            RightDataSet.append(Data)
    return LeftDataSet, RightDataSet


#  获取样本特征值的矩阵
def ObtainFeatureValueMat(DataSet):
    dataSet = np.array(DataSet)
    featureNum = len(FEATURE) # 获取样本所有特征数目
    featureMat = []
    for i in range(featureNum):
        featureVec = list(set(dataSet[:, i]))
        featureVec = [float(item) for item in featureVec]
        featureMat.append(featureVec)
    return featureMat


####################################################### 回归树生成部分 ###################################################

#  数据集误差计算函数 -- 计算数据集的方差并乘以数据集的个数
def ComputeVar(DataSet):
    '''This function is to compute the error of data set!'''
    dataSet = np.array(DataSet)
    Len = len(dataSet)
    if Len > 0: return np.var(dataSet[:,-1]) * Len
    else: return 0.0


# 叶子节点生成函数 -- 计算数据集均值
def regLeaf(DataSet):
    return np.mean(DataSet[:,-1])


#  选择最优特征以及所对应的特征值 -- 参数：数据集，特征值集，叶节点形式（均值or其他），误差计算方式，用户自定义参数
def ChooseBestFeatureAndValue(DataSet, featureMat, leafType = regLeaf, errType = ComputeVar, ops=(1,4)):
    '''This function is to choose the best Splited feature and the value!'''
    dataSet = np.array(DataSet)
    tolS = ops[0]
    tolN = ops[1]
    # 数据集的取值都相等，则停止分割
    if len(set(dataSet[:, -1])) == 1:
        return None, regLeaf(dataSet)
    # 否则，继续进行数据分割分析
    S = errType(DataSet) # 当前数据集的误差
    featureNum = len(FEATURE)  # 获取样本所有特征数目
    BestS = S # 记录数据集分割后的最小误差
    DataErrorVec = []
    BestFeatureAndValue = {} # 以字典的形式记录每个特征所对应的最佳切分点的值
    for feature in FEATURE:
        for value in featureMat[FEATURE.index(feature)]:
            LeftDataSet, RightDataSet = SplitDataSet(DataSet, feature, value)
            LeftDataSetError = errType(LeftDataSet)
            RightDataSetError = errType(RightDataSet)
            DataErrorVec.append(LeftDataSetError+RightDataSetError)
        BestFeatureAndValue[feature] = featureMat[FEATURE.index(feature)][DataErrorVec.index(min(DataErrorVec))]
        if BestS > min(DataErrorVec): BestS = min(DataErrorVec)
    BestFeatureAndValue = sorted(BestFeatureAndValue.iteritems(), key = operator.itemgetter(1), reverse = True)[0]
    # 如果分割后的数据集误差与不分割的数据集误差在很小，则不需要继续分割
    if (S - BestS)<tolS:
        return None, leafType(dataSet)
    # 如果切割后的数据集规模过于小，则停止分割
    LeftDataSet, RightDataSet = SplitDataSet(DataSet, BestFeatureAndValue[0], BestFeatureAndValue[1]) # 按照最优特征及其对应的值进行分割数据得到子集
    if (len(LeftDataSet) < tolN) or (len(RightDataSet) < tolN):
        return None, leafType(dataSet)
    # 如果上述停止条件均不满足，则返回最优特征及其值，作为回归树的判断节点
    return BestFeatureAndValue[0], BestFeatureAndValue[1]


# 递归构造回归树
def createTree(DataSet, featureMat, leafType = regLeaf, errType = ComputeVar, ops = (1,4)):
    BestFeature, BestValue = ChooseBestFeatureAndValue(DataSet, featureMat, leafType, errType, ops)
    if BestFeature == None:  # 如果无特征返回，表明数据集分割停止，应当生成叶子节点
        return BestValue
    retTree = {}
    retTree['spInd'] = BestFeature
    retTree['spVal'] = BestValue
    LeftDataSet, RightDataSet = SplitDataSet(DataSet, BestFeature, BestValue)
    retTree['left'] = createTree(LeftDataSet, featureMat, leafType, errType, ops)
    retTree['right'] = createTree(RightDataSet, featureMat, leafType, errType, ops)
    return retTree

# DataSet = LoadDataSet('ex0.txt')
# dataSet =[]
# for Line in DataSet:
#     Line = Line[1:]
#     dataSet.append(Line)
# featureMat = ObtainFeatureValueMat(dataSet)
# print createTree(dataSet, featureMat)