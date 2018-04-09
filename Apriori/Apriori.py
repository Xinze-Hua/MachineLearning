# -*- coding: utf-8 -*-
from numpy import *


# 生成样本数据
def CreatData():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5],[2, 5]]


# 导入样本数据
def LoadData(filename):
    labelMat = []
    dataMat = []
    fr = open(filename)
    dataLines = fr.readlines()
    # print dataLines
    for line in dataLines:
        lineArr = line.strip().split()
        tempArr = []
        for i in range(0, len(lineArr)):
            tempArr.append(float(lineArr[i]))
        dataMat.append(tempArr)
        labelMat.append(float(lineArr[0]))
    return dataMat, labelMat



# 生成初始的频繁项候选集
def creatC1(dataSet):
    C1 = []
    for transaction in dataSet:  # 遍历数据集中的每一项
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)


# 扫描频繁项候选集找到其中满足条件的频繁项集
def ScanD(dataSet, Ck, minSupport):
    ssCnt = {} # 用来存储频繁集以及对应出现的次数
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = len(dataSet)
    retList = []  # 用来存储满足条件的频繁项集
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/float(numItems)
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return supportData, retList



# 生成新的频繁项候选集
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)  # k阶频繁集的个数
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2] # 访问的是0到k-3之间的元素
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if(L1 == L2):
                retList.append(Lk[i]|Lk[j])
    return retList


# 遍历所有数据集生成各阶的频繁集
def apriori(dataSet, minSupport = 0.5):
    C1 = creatC1(dataSet)
    supportData, retList = ScanD(dataSet, C1, minSupport)
    L = [retList]
    k = 2
    while(len(L[k-2]) > 1):
        Ck = aprioriGen(retList, k)
        supData, retList = ScanD(dataSet, Ck, minSupport)
        supportData.update(supData)
        L.append(retList)
        k += 1
    return L, supportData


# 生成关联规则函数
def generateRule(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                ruleFromConseq(L, H1, freqSet, supportData, bigRuleList, minConf)
            else:  # 对于元素个数小于等于2的频繁项，直接进行关联规则生成
                calConf(L, H1, freqSet, supportData, bigRuleList, minConf)


def calConf(L, H, freqSet, supportData, bigRuleList, minConf):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet - conseq]
        if conf >= minConf:
            print freqSet - conseq, "-->", conseq, "conf:", conf
            bigRuleList.append([freqSet - conseq, conseq, conf])
            prunedH.append(conseq)
    return prunedH

def ruleFromConseq(L, H, freqSet, supportData, bigRuleList, minConf):
    m = len(H[0])
    if(len(freqSet) > m + 1):
        Hm = aprioriGen(H, m + 1)  # 生成新的子集
        Hm = calConf(L, Hm, freqSet, supportData, bigRuleList, minConf)
        if(len(Hm) > 1):
            ruleFromConseq(L, Hm, freqSet, supportData, bigRuleList, minConf)  # 这里运用了递归的算法


######################################################## 建立关联规则 ####################################################
# dataSet = CreatData()
# L, supportData = apriori(dataSet)
# bigRuleList = generateRule(L, supportData, 0.6)


######################################################## 毒蘑菇检测 #####################################################
# dataMat, labelMat = LoadData("mushroom.dat")
# poisonous_mushroom_Data = array(dataMat)[nonzero(array(labelMat)[:] == 2)[0],:]
# L, supportData = apriori(dataMat, 0.3)
# for item in  L[1]:
#     if 2.0 in item: print item
# for item in  L[3]:
#     if 2.0 in item: print item
# for item in  L[4]:
#     if 2.0 in item: print item
