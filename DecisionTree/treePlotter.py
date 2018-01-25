# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import math


decisionNode = dict(boxstyle = 'sawtooth', fc = '0.8')  # 设置决策节点的属性,存储为字典格式
leafNode = dict(boxstyle = 'round4', fc = '0.8')  # 设置叶子节点的属性,存储为字典格式
arrow_args = dict(arrowstyle = '<-')  # 设置箭头的属性,存储为字典格式


# 节点绘制函数
def plotNode(NodeText, centerPt, parentPt, NodeType):   # 添加注释函数
    createPlot.ax1.annotate(NodeText, xy = parentPt, xycoords = 'axes fraction', xytext = centerPt, textcoords = 'axes fraction', va = 'center', ha = 'center', bbox = NodeType, arrowprops = arrow_args)


# 计算决策树的深度
def ComputeTreeDepth(MyTree):  # 计算树的深度
    MaxDepth = 0 # 用来记录树的最大深度
    firstStr = MyTree.keys()[0] # 节点的key
    secondDict = MyTree[firstStr] # 节点的Value
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': #  如果节点值为字典类型，则表示树的深度加1
            thisDepth = 1 + ComputeTreeDepth(secondDict[key])  # 进行递归
        else: thisDepth = 1  # 遇到叶子节点就停止
        if thisDepth>MaxDepth: MaxDepth=thisDepth
    return MaxDepth


# 统计叶子节点的数目
def CountLeafNode(MyTree):  # 计算树的宽度
    numberLeafs = 0.0  # 用来记录叶子节点的数目
    firstStr = MyTree.keys()[0]
    secondDict = MyTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': # 如果节点值为字典类型，继续进行递归计算
            numberLeafs += CountLeafNode(secondDict[key])
        else: numberLeafs += 1  # 如果节点值不为字典类型，则为叶子节点，则累加
    return numberLeafs


# 在箭头中间添加文字说明
def plotMidText(centerPt, parentPt, txtString):  # 在注释中间添加文字说明
    xMid = (centerPt[0]-parentPt[0])/2.0 + parentPt[0]
    yMid = (centerPt[1]-parentPt[1])/2.0 + parentPt[1]
    createPlot.ax1.text = (xMid, yMid, txtString)


# 建立绘图
def createPlot(MyTree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = []) # 除去坐标轴
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(CountLeafNode(MyTree)) # 获取树的宽度，即叶子节点的个数
    plotTree.totalD = float(ComputeTreeDepth(MyTree))  # 获取树的深度
    plotTree.xOff = -0.5 / plotTree.totalW  # X轴的偏移量
    # print plotTree.xOff
    plotTree.yOff = 1.0  # Y轴的偏移量
    plotTree(MyTree, (0.5, 1.0), '')
    plt.show()


# 绘制树形图
def plotTree(MyTree, parentPt, nodeTxt):  # 绘制树形图
    numLeafs = CountLeafNode(MyTree)  # 当前子树的叶子数目
    depth = ComputeTreeDepth(MyTree)   # 当前子树的深度
    firstStr = MyTree.keys()[0]  # 当前子树的根节点名称
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW , plotTree.yOff)  # 依据当前子树的叶子节点的个数确定中心节点
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 在图中画出节点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  # 更新Y轴偏移量
    secondDict = MyTree[firstStr]  # 当前子树的子树
    for key in secondDict.keys():  # 对当前子树的子节点进行判断处理
        if type(secondDict[key]).__name__ == 'dict':  # 如果子节点也是子树的话，重复递归构造
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW  # 每画一个叶子节点，就更新X轴的偏移量
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD  # 更新Y轴偏移量

#
# # MyTree = {'House': {'1': 'yes', '0': {'Work': {'1': 'yes', '0': 'no'}}}}
# MyTree = {'No Surfacing': {'1': {'flippers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}
# createPlot(MyTree)