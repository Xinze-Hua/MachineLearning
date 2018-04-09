# -*- coding: utf-8 -*-
from numpy import *


# ******************************************************* 1.1 生成数据 **************************************************
# 生成数据
def loadData():
    dataSet = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return dataSet

# 数据处理
def DataProcess(dataSet):
    D = {}
    for item in dataSet:
        D[frozenset(item)] = 1
    return D


# ******************************************************* 1.2 构建FP树 **************************************************
# 声明FP树的节点类
class FPTreeNode:
    def __init__(self, name, count, parent):
        self.Name = name
        self.Count = count
        self.Parent = parent
        self.NodeLink = None
        self.Children = {}

    def inc(self, num):
        self.Count += num

    def disp(self, i = 0):
        print " "*i, self.Name, ' ', self.Count
        for child in self.Children.values():
            child.disp(i+1)


# 生成FP树
def CreatFPTree(DataSet, minSup = 1):
    HeadTable = {}
    for trans in DataSet.items():  # 第一次遍历整个数据集，建立表头
        for item in trans[0]:
            HeadTable[item] = HeadTable.get(item, 0) + DataSet[trans[0]]
    for el in HeadTable.keys():  # 剔除不满足最小支持度要求的元素项
        if HeadTable[el] < minSup:
            del(HeadTable[el])
    if(len(HeadTable) == 0):
        return None, None
    freqItemSet = set(HeadTable.keys())
    retTree = FPTreeNode("NuLL Set", 1, None)
    for k in HeadTable:
        HeadTable[k] = [HeadTable[k], None]
    for trans, count in DataSet.items(): # 第二次遍历整个数据集，递归生成FP树
        localD = {}
        for item in trans:
            if item in freqItemSet:
                localD[item] = HeadTable[item][0]
        if len(localD) > 0:
            orderedItemSet = [v[0] for v in sorted(localD.items(), key = lambda p:p[1], reverse=True)]
            UpdateTree(orderedItemSet, retTree, HeadTable, count)
    return retTree, HeadTable


# 更新树的节点
def UpdateTree(orderedItemSet, retTree, HeadTable, count):
    if orderedItemSet[0] in retTree.Children:
        retTree.Children[orderedItemSet[0]].inc(count)
    else:
        retTree.Children[orderedItemSet[0]] = FPTreeNode(orderedItemSet[0], count, retTree)
        if HeadTable[orderedItemSet[0]][1] == None:
            HeadTable[orderedItemSet[0]][1] = retTree.Children[orderedItemSet[0]]
        else:
            UpdateHeadTable(HeadTable[orderedItemSet[0]][1], retTree.Children[orderedItemSet[0]], count)
    if(len(orderedItemSet) > 1):
        UpdateTree(orderedItemSet[1::], retTree.Children[orderedItemSet[0]], HeadTable, count)


# 更新表头
def UpdateHeadTable(head, newNode, count):
    while (head.NodeLink != None):
        head = head.NodeLink
    head.NodeLink = newNode


# **************************************************** 1.3 从FP树中进行挖掘 **********************************************
# 寻找所有表头元素所对应的条件基
def findPreFixPath(basePat, tabelNode):
    condPats = {}
    while tabelNode!= None:
        preFixPath = []
        ascendTree(preFixPath, tabelNode)
        if len(preFixPath) > 1:
            condPats[frozenset(preFixPath[1:])] = tabelNode.Count
        tabelNode = tabelNode.NodeLink
    return condPats


# 查找对应元素的条件基
def ascendTree(preFixPath, tabelNode):
    if tabelNode.Parent != None:
        preFixPath.append(tabelNode.Name)
        ascendTree(preFixPath, tabelNode.Parent)


# 对每一个表头元素进行频繁集挖掘
def MineTree(retTree, HeadTable, minSup, freqItemList, preFix):
    bigL = [v[0] for v in sorted(HeadTable.items(), key = lambda p:p[1])]
    for item in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(item)
        freqItemList.append(newFreqSet)
        condPats = findPreFixPath(item, HeadTable[item][1])
        CondTree, CondHeadTable = CreatFPTree(condPats, minSup)
        if CondTree != None:
            MineTree(CondTree, CondHeadTable, minSup, freqItemList, newFreqSet)


# ******************************************************* 1.4 算法执行 **************************************************

dataSet = [line.split() for line in open('kosarak.dat').readlines()]
DataSet = DataProcess(dataSet)
retTree, HeadTable = CreatFPTree(DataSet, minSup = 100000)
freqItemList = []
MineTree(retTree, HeadTable, 100000, freqItemList, set([]))
print freqItemList
