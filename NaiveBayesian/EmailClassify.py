# -*- coding:utf-8 -*-
from numpy import *
from os import listdir
import re  #  正则表达式
import feedparser


# 样本单词提取
def SpiltText(fileList):
    regEx = re.compile('\\W*')  # 以所有非数字字母作为分隔符进行单词文本分割
    WordList = []
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


# 建立词汇表
def creatVocabList(WordList):
    Vocabset = set([])
    for doucument in WordList:
        Vocabset = Vocabset | set(doucument)
    return list(Vocabset)


# 将字符向量通过字符表转化为数字向量
def setOfWordToVector(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  #  这里使用的是词袋模型
        # else: print "the word ", word, "is not in my vocabulary!"
    return returnVec


# 先验概率、条件概率训练
def trainNaiveBayesian(trainMartix, trainCategory):
    docNum = len(trainCategory)
    numTrainMatrix = len(trainMartix)
    pAbusive = sum(trainCategory) / float(docNum)  # 计算先验概率
    pNormal = 1.0 - pAbusive
    numWords = len(trainMartix[0])
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainMatrix):
        if(trainCategory[i] == 1):  # 类别1
            p1Num += trainMartix[i]
            p1Denom += sum(trainMartix[i])
        else:                       # 类别2
            p0Num += trainMartix[i]
            p0Denom += sum(trainMartix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 贝叶斯分类器
def ClassifyNaiveBayesian(vecToClassify, p0vec, p1vec, pClass):
    p1 = sum(vecToClassify * p1vec) + log(pClass)
    p0 = sum(vecToClassify * p0vec) + log(1-pClass)
    if(p0 < p1):
        return 1
    else:
        return 0


################################################ 网站恶意留言识别实例 ####################################################
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


################################################### 邮件分类实例 ########################################################
# 建立分类模型
def ClassifyEmail():
    HamemailList = ['email/ham/' + el for el in listdir('email/ham')]  # 正常邮件的文件名列表
    SpamemailList = ['email/spam/' + el for el in listdir('email/spam')]  # 垃圾邮件的文件名列表
    HamWordList = SpiltText(HamemailList)  #  提取所有正常邮件的单词列表
    SpamWordList = SpiltText(SpamemailList)  #  提取所有垃圾邮件的单词列表
    HamEmailClassVec = zeros(len(HamWordList))  # 正常邮件的类别向量 0表示
    SpamEmailClassVec = ones(len(SpamWordList))  # 垃圾邮件的类别向量 1表示
    WordList = SpamWordList + HamWordList  # 所有邮件的单词列表
    EmailClassVec = hstack((SpamEmailClassVec, HamEmailClassVec))  # 邮件类别向量
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
        if(ClassifyNaiveBayesian(testMartix[i], p0vec, p1vec, pClass) != testClassVec[i]):
            errorCount += 1
    return errorCount/10.0


# 进行10次随机验证，获取分类错误率
def ClassifyTest():
    sum = 0.0
    for i in range(10):
        p = ClassifyEmail()
        sum += p
    print sum/10.0

ClassifyTest()


############################################### 从个人广告中获取区域倾向实例 ##############################################
# 文本切割
def textParse(wordText):
    regEx = re.compile('\\W*')
    listofToken = regEx.split(wordText)
    return [word.lower() for word in listofToken if len(word) > 2]


# 计算单词出现频率并选择出现频率最高的三十个单词
def calMostFreq(vocabList, fullText):
    import operator
    freDict = {}  #  字典类型
    for token in vocabList:
        freDict[token] = fullText.count(token)
    SortedfreDict = sorted(freDict.iteritems(), key = operator.itemgetter(1), reverse = True)
    return SortedfreDict[:30]


# 建立分类模型
def localWord(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minlen = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(minlen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)  #  将wordlist整体作为一个元素追加到docList
        fullText.extend(wordList)  #  将wordlist里面的元素合并到fulltext
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)  #  将wordlist整体作为一个元素追加到docList
        fullText.extend(wordList)  #  将wordlist里面的元素合并到fulltext
        classList.append(0)
    FullText = [el.encode('unicode-escape').decode('string_escape') for el in fullText]  # 将Unicode编码转换为str
    vocubList = creatVocabList(docList)  # 创建词汇表
    fr = open('stopWords.txt')
    StopWordList = fr.readlines()
    StopWordMat = []
    for wordLine in StopWordList:
        StopWordMat.extend(textParse(wordLine))
    # StopWordList = [StopWord.strip('\n') for StopWord in StopWordList]
    for word in StopWordMat:
        if word in vocubList:
            vocubList.remove(word)
    # top30Words = calMostFreq(vocubList, FullText) # 获取频率最高的30个单词
    # for word in top30Words:  #  top30Words是字典类型的数据，键为单词，对应值为频率
    #     if word[0] in vocubList:
    #         vocubList.remove(word[0])  #  去掉词汇表中出现频率最高的30个单词
    trainingSet = range(2*minlen)
    testingSet = []
    for i in range(5):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testingSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWordToVector(vocubList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0vec, p1vec, pClass = trainNaiveBayesian(trainMat, trainClass)
    errorCount = 0
    for docIndex in testingSet:
        wordvector = setOfWordToVector(vocubList, docList[docIndex])
        if(ClassifyNaiveBayesian(wordvector, p0vec, p1vec, pClass) != classList[docIndex]):
            errorCount += 1
    return errorCount / 6.0, p0vec, p1vec, pClass, vocubList


# 获取最具表征性的单词
def getTopWord(p0vec, p1vec, pClass, vocubList):
    import operator
    topNasa = []; topRocket = []
    for i in range(len(p0vec)):
        if p0vec[i] > -4.4 :topRocket.append((vocubList[i], p0vec[i]))
    sortedTopRocket = sorted(topRocket, key = lambda pair:pair[1], reverse = True)
    print "#################Rocket################"
    for word in sortedTopRocket:
        print word[0]

    for i in range(len(p1vec)):
        if p1vec[i] > -4.0 : topNasa.append((vocubList[i], p1vec[i]))
    sortedTopNasa = sorted(topNasa, key=lambda pair: pair[1], reverse=True)
    print "##################Nasa#################"
    for word in sortedTopNasa:
        print word[0]
feed0 = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
feed1 = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
errorRate, p0vec, p1vec, pClass, vocubList = localWord(feed1, feed0)
print 'the error rate is ', errorRate
getTopWord(p0vec, p1vec, pClass, vocubList)
