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
    if(p0<p1):
        # print "正常邮件"
        return 0
    else:
        # print "垃圾邮件"
        return 1


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
            listOfToken = [el.lower() for el in listOfToken]
            Word_afetrDeal += listOfToken
        WordList.append(Word_afetrDeal)
    return WordList


def ClassifyEmail():
    HamemailList = ['email/ham/' + el for el in listdir('email/ham')]  # 正常邮件的文件名列表
    SpamemailList = ['email/spam/' + el for el in listdir('email/spam')]  # 垃圾邮件的文件名列表
    HamWordList = SpiltText(HamemailList)  #  提取所有正常邮件的单词列表
    SpamWordList = SpiltText(SpamemailList)  #  提取所有垃圾邮件的单词列表
    HamEmailClassVec = zeros(len(HamWordList))  # 正常邮件的类别向量
    SpamEmailClassVec = ones(len(SpamWordList))  # 垃圾邮件的类别向量
    WordList = SpamWordList + HamWordList  # 所有邮件的单词列表
    EmailClassVec = hstack((HamEmailClassVec, SpamEmailClassVec))  # 邮件类别向量
    EmailClassList = EmailClassVec.tolist()  # 所有邮件的类别列表
    VocabList = creatVocabList(WordList)  # 建立词汇表
    testVec = []  # 测试邮件的单词向量
    trainMartix = []  # 训练集
    testMartix = []  # 测试集
    testClassVec = []  # 测试集的类别
    for i in range(10):
        randIndex = int(random.uniform(0, len(WordList)))  # 在0-50之间产生10个随机数
        testVec.append(WordList[randIndex])  # 获取10个随机测试邮件
        testClassVec.append(EmailClassList[randIndex])  # 10个测试邮件的类别向量
        del(WordList[randIndex])  # 获取40个训练邮件
        del(EmailClassList[randIndex])  # 40个训练集的类别列表
    for el in WordList:
        trainMartix.append(setOfWordToVector(VocabList, el))  # 将训练集转化成数字向量形式
    for el in testVec:
        testMartix.append(setOfWordToVector(VocabList, el))   # 将测试集转化成数字向量形式
    p0vec, p1vec, pClass = trainNaiveBayesian(trainMartix, EmailClassList)   # 计算先验概率，条件概率等
    errorCount = 0.0
    for i in range(10):
        if(ClassifyNaiveBayesian(testMartix[i], p0vec, p1vec, pClass)==testClassVec[i]):
            errorCount += 1
    return errorCount/10.0

def ClassifyTest():
    sum = 0.0
    for i in range(10):
        p = ClassifyEmail()
        sum += p
    print sum/10.0

ClassifyTest()




















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