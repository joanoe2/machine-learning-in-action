import shannon

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    dataSet = []
    for line in lines:
        line = line.strip()
        if line:
            dataSet.append(line.split('\t'))
    return dataSet

dataSet = file2matrix('data.txt')
labels = ['age', 'prescript', 'astigmatic', 'tearRate']

lensesTree = shannon.createTree(dataSet, labels)
print(lensesTree)


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

featLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# presbyopic	myope	yes	normal	hard
testVec = ['presbyopic', 'myope', 'yes', 'normal']
result = classify(lensesTree, featLabels, testVec)
print(result)