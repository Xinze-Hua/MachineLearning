# -*- coding:utf-8 -*-
from numpy import *
from os import listdir
import re  #  正则表达式

# 训练实例
def loadDataSet():
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

# 建立词汇表
def creatVocabList(dataList):
    Vocabset = set([])
    for doucument in dataList:
        Vocabset = Vocabset | set(doucument)
    return list(Vocabset)


# 将实例通过字符表转化为向量
def setOfWordToVector(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  #  这里使用的是词袋模型
        else: print "the word ", word, "is not in my vocabulary!"
    return returnVec


# 概率训练
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
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 贝叶斯分类器
def ClassifyNaiveBayesian(vecToClassify, p0vec, p1vec, pClass):
    p1 = sum(vecToClassify*p1vec) + log(pClass)
    p0 = sum(vecToClassify * p0vec) + log(1-pClass)
    if(p0>p1):
        print "正常邮件"
    else: print "垃圾邮件"


# 样本邮件单词提取
def SpiltText(fileList):
    regEx = re.compile('\\W*')
    WordList = []
    i = 0
    for filename in fileList:
        fr= open(filename)
        wordList = fr.readlines()
        Word_afetrDeal = []
        for wordLine in wordList:
            listOfToken = regEx.split(wordLine)
            listOfToken = filter(lambda x: len(x) >= 3 and (not x.isdigit()) , listOfToken)
            Word_afetrDeal += listOfToken
        WordList.append(Word_afetrDeal)
    return WordList

# 获得词汇表
hamemailList = ['email/ham/' + el for el in listdir('email/ham')]
SpamemailList = ['email/spam/' + el for el in listdir('email/spam')]

hamWord = SpiltText(HamemailList)
spamWordSet = SpiltText(SpamemailList)
# VocubSet = spamWordSet|hamWordSet
# VocubSet = [el.lower() for el in VocubSet]





















# ListofPost, ListClasses = loadDataSet()  #  获得实例数据
# VocabList = creatVocabList(ListofPost)   #  根据实例数据生成字典
# trainMat = []
# for postinDoc in ListofPost:
#     trainMat.append(setOfWordToVector(VocabList, postinDoc))
# p0Vect, p1Vect, pClass = trainNaiveBayesian(trainMat, ListClasses)
# temp = setOfWordToVector(VocabList, ['stupid', 'love', 'dalmation'])
# ClassifyNaiveBayesian(temp, p0Vect, p1Vect, pClass)
#
# def SplitEmailtext(filename):
#     regEx = re.compile('\\W*')
#     fr = open(filename)
#     wordlist = fr.readlines()
#     for wordline in wordlist:
#         temp = regEx.split(wordline)
#         print temp
#
# SplitEmailtext('email/spam/1.txt')