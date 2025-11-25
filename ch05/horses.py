import math
import random

import numpy as np


def sigmoid(inX):
    """Numerically stable sigmoid for scalars or numpy arrays.

    math.exp(-x) overflows when x is a large negative number (so -x is large positive).
    To avoid that, we compute the result with different formulas depending on the sign of x:
    - if x >= 0: 1 / (1 + exp(-x))
    - else: exp(x) / (1 + exp(x))

    This works for both numpy arrays and scalars and avoids OverflowError.
    """
    # Use numpy for arrays, but also support scalars
    x = np.asarray(inX, dtype=float)
    # For scalars, use math to preserve scalar result and avoid array operations
    if x.ndim == 0:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            return math.exp(x) / (1.0 + math.exp(x))

    # For arrays, only compute exp on the required slices to avoid overflow
    out = np.empty_like(x)
    pos_mask = x >= 0
    # For positive entries: 1 / (1 + exp(-x)) (safe because -x is <= 0)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    # For negative entries: exp(x) / (1 + exp(x)) (safe because x <= 0)
    out[~pos_mask] = np.exp(x[~pos_mask]) / (1.0 + np.exp(x[~pos_mask]))
    return out


def classifyVector(inX, weights):
    prob = sigmoid(np.dot(inX, weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4.0 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.dot(dataMatrix[randIndex], weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights


def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")

    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainingWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainingWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print(
        "after %d iterations the average error rate is: %f"
        % (numTests, errorSum / float(numTests))
    )


if __name__ == "__main__":
    multiTest()
