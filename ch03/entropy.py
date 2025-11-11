from math import log


def entropy(dataSet):
    print("Calculating entropy for the dataset...")

    labels = [row[-1] for row in dataSet]
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    entropy = 0.0
    total = len(dataSet)
    for c in counts.values():
        prob = c / total
        entropy -= prob * log(prob, 2)

    return entropy


def splitDataSet(dataSet, axis, value):
    print(f"Splitting dataset on axis {axis} for value {value}...")
    return [row[:axis] + row[axis + 1 :] for row in dataSet if row[axis] == value]


def chooseBestFeatureToSplit(dataSet):
    print("Choosing the best feature to split on...")
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = entropy(dataSet)
    bestGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        values = set(row[i] for row in dataSet)
        newEntropy = 0.0
        for v in values:
            subset = splitDataSet(dataSet, i, v)
            weight = len(subset) / len(dataSet)
            newEntropy += weight * entropy(subset)
        gain = baseEntropy - newEntropy
        if gain > bestGain:
            bestGain = gain
            bestFeature = i

    return bestFeature

def createTree(dataSet, featureNames):
    print("Creating decision tree...")
    labels = [row[-1] for row in dataSet]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(dataSet[0]) == 1:
        return max(set(labels), key=labels.count)

    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureName = featureNames[bestFeature]
    tree = {bestFeatureName: {}}
    del featureNames[bestFeature]

    values = set(row[bestFeature] for row in dataSet)
    for v in values:
        subset = splitDataSet(dataSet, bestFeature, v)
        subtree = createTree(subset, featureNames[:])
        tree[bestFeatureName][v] = subtree

    return tree