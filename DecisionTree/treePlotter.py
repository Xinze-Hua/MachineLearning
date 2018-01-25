# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import math


decisionNode = dict(boxstyle = 'sawtooth', fc = '0.8')  # 设置决策节点的属性,存储为字典格式
leafNode = dict(boxstyle = 'round4', fc = '0.8')  # 设置叶子节点的属性,存储为字典格式
arrow_args = dict(arrowstyle = '<-')  # 设置箭头的属性,存储为字典格式


def plotNode(NodeText, centerPt, parentPt, NodeType):
    createPlot.ax1.annotate(NodeText, xy = parentPt, xycoords = 'axes fraction', xytext = centerPt, textcoords = 'axes fraction', va = 'center', ha = 'center', bbox = NodeType, arrowprops = arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor = 'white')  # 第一个参数表示图ID，第二个参数表示背景颜色，通过plt.figure.__doc__了解更多关于figure的参数详情
    fig.clf() # 清除图像内容
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111,frameon = False, **axprops)  # 定义一个函数参数，该变量是一个全局变量
    plotNode('DecisionNode', (0.5, 1), (0.5, 1), decisionNode)
    plotNode('LeafNode', (0.8,0.1), (0.5,1), leafNode)
    plt.show()




createPlot()
