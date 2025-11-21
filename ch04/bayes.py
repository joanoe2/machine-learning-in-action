from numpy import array, log, ones


def loadDataset():
    postingLit = [
        ["my", "dog", "has", "flea", "problems", "help", "please"],
        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        ["stop", "posting", "stupid", "worthless", "garbage"],
        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
        ["quit", "buying", "worthless", "dog", "food", "stupid"],
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingLit, classVec


def createVocabList(dataSet):
    vocabList = set()
    for doc in dataSet:
        vocabList = vocabList | set(doc)
    return sorted(list(vocabList))


def setOfWords2Vec(vocabList, inputSet):
    returnMat = [0] * len(vocabList)
    for w in inputSet:
        if w in vocabList:
            returnMat[vocabList.index(w)] = 1
        else:
            print(f"The word: {w} is not in my Vocabulary!")
    return returnMat


def trainNB(trainMatrix, trainCat):
    numTrainDocs = len(trainMatrix)
    numOfWords = len(trainMatrix[0])
    pAbusive = sum(trainCat) / float(numTrainDocs)
    p0Num = ones(numOfWords)
    p1Num = ones(numOfWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCat[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    prob1 = log(p1Num / p1Denom)
    prob0 = log(p0Num / p0Denom)

    return prob0, prob1, pAbusive


def classifyNB(vec2Classify, prob0, prob1, pCat):
    p1 = sum(vec2Classify * prob1) + log(pCat)
    p0 = sum(vec2Classify * prob0) + log(1.0 - pCat)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    postingList, classVec = loadDataset()
    myVocabList = createVocabList(postingList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB(trainMat, classVec)
    testEntry = ["dalmation", "cute"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ["my", "stupid", "dog"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
