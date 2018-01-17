# -*- coding:utf-8 -*-
from numpy import *

def loadDataSet():  # 训练实例
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def creatVocabList(dataSet):  # 建立字符表
    Vocabset = set([])
    for doucument in dataSet:
        Vocabset = Vocabset | set(doucument)
    return list(Vocabset)


def setOfWordToVector(vocabList, inputSet):  # 将实例通过字符表转化为向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word ", word, "is not in my vocabulary!"
    return returnVec


def trainNaiveBayesian(trainMartix, trainCategory):
    docNum = len(trainCategory)
    numTrainMatrix = len(trainMartix)
    pAbusive = sum(trainCategory) / float(docNum)  # 计算侮辱性留言的概率
    pNormal = 1.0 - pAbusive  # 计算正常留言的概率
    numWords = len(trainMartix[0])
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainMatrix):
        if(trainCategory[i] == 1):  # 垃圾邮件
            p1Num += trainMartix[i]
            p1Denom += sum(trainMartix[i])
        else:                       # 正常邮件
            p0Num += trainMartix[i]
            p0Denom += sum(trainMartix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p1Vect, p0Vect

ListofPost, ListClasses = loadDataSet()  #  获得实例数据
VocabList = creatVocabList(ListofPost)   #  根据实例数据生成字典
trainMat = []
for postinDoc in ListofPost:
    trainMat.append(setOfWordToVector(VocabList, postinDoc))
p1Vect, p0Vect = trainNaiveBayesian(trainMat, ListClasses)
print p0Vect
print p1Vect
print "ok"
