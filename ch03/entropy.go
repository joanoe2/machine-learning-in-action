package main

import (
	"fmt"
	"math"
)

func calculateEntropy(dataSet [][]string) float64 {
	labelCounts := make(map[string]int)
	for _, vector := range dataSet {
		label := vector[len(vector)-1]
		labelCounts[label]++
	}

	var entropy float64
	totalEntries := float64(len(dataSet))
	for _, count := range labelCounts {
		p := float64(count) / totalEntries
		entropy -= p * math.Log2(p)
	}
	return entropy
}

func splitDataSet(dataSet [][]string, featureIndex int, value string) [][]string {
	var subset [][]string
	for _, vec := range dataSet {
		if vec[featureIndex] == value {
			reducedVec := append(vec[:featureIndex], vec[featureIndex+1:]...)
			subset = append(subset, reducedVec)
		}
	}
	return subset
}

func chooseBestFeature(dataSet: [][]string) int {
	numFeatures := len(dataSet[0]) - 1
	baseEntropy := calculateEntropy(dataSet)
	bestInfoGain := 0.0
	bestFeature := -1

	for i := 0; i < numFeatures; i++ {
		featureValues := make(map[string]bool)
		for _, vector := range dataSet {
			featureValues[vector[i]] = true
		}
		var newEntropy float64
		for value := range featureValues {
			subset := splitDataSet(dataSet, i, value)
			p := float64(len(subset)) / float64(len(dataSet))
			newEntropy += p * calculateEntropy(subset)
		}
		infoGain := baseEntropy - newEntropy
		if infoGain > bestInfoGain {
			bestInfoGain = infoGain
			bestFeature = i
		}
	}
	return bestFeature
}

func createTree(dataSet [][]string, featureLabels []string) map[string]interface{} {
	classList := make([]string, len(dataSet))
	for i, vector := range dataSet {
		classList[i] = vector[len(vector)-1]
	}

	allSame := true
	for i := 1; i < len(classList); i++ {
		if classList[i] != classList[0] {
			allSame = false
			break
		}
	}
	if allSame {
		return map[string]interface{}{"label": classList[0]}
	}

	if len(dataSet[0]) == 1 {
		labelCounts := make(map[string]int)
		for _, label := range classList {
			labelCounts[label]++
		}
		var majorityLabel string
		maxCount := 0
		for label, count := range labelCounts {
			if count > maxCount {
				maxCount = count
				majorityLabel = label
			}
		}
		return map[string]interface{}{"label": majorityLabel}
	}
	
	bestFeature := chooseBestFeature(dataSet)
	bestFeatureLabel := featureLabels[bestFeature]
	tree := make(map[string]interface{})
	tree[bestFeatureLabel] = make(map[string]interface{})
	featureValues := make(map[string]bool)
	for _, vector := range dataSet {
		featureValues[vector[bestFeature]] = true
	}

	for value := range featureValues {
		subLabels := append([]string{}, featureLabels...)
		subLabels = append(subLabels[:bestFeature], subLabels[bestFeature+1:]...)
		subDataSet := splitDataSet(dataSet, bestFeature, value)
		tree[bestFeatureLabel].(map[string]interface{})[value] = createTree(subDataSet, subLabels)
	}
	return tree
}

func classify(tree map[string]interface{}, featureLabels []string, sample []string) string {
	rootFeature := ""
	for k := range tree {
		rootFeature = k
		break
	}
	featureIndex := -1
	for i, label := range featureLabels {
		if label == rootFeature {
			featureIndex = i
			break
		}
	}
	subTree := tree[rootFeature].(map[string]interface{})
	sampleValue := sample[featureIndex]
	var result string
	if nextNode, ok := subTree[sampleValue]; ok {
		switch nextNode.(type) {
		case map[string]interface{}:
			result = classify(nextNode.(map[string]interface{}), append(featureLabels[:featureIndex], featureLabels[featureIndex+1:]...), append(sample[:featureIndex], sample[featureIndex+1:]...))
		case string:
			result = nextNode.(string)
		}
	} else {
		result = "Unknown"
	}
	return result
}