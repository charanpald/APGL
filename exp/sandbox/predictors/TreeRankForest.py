
"""
A Python implementation of ranking forests using TreeRank.
"""
import numpy
import logging
import scikits.learn.cross_val as cross_val
from apgl.metabolomics.AbstractTreeRank import AbstractTreeRank
from apgl.metabolomics.TreeRank import TreeRank
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util
from apgl.util.Evaluator import Evaluator

class TreeRankForest(AbstractTreeRank):
    def __init__(self, generateLeafRank):
        """
        Create a new TreeRankForest object and initialise with a function that
        generates leaf rank objects, for example LinearSVM or DecisionTree. The
        left node is more positive than the right for each split.

        :param generateLeafRank: A method to create leaf rank classifiers. 
        """
        super(TreeRankForest, self).__init__(generateLeafRank)
        self.numTrees = 5
        self.sampleSize = 0.5
        self.sampleReplace = True

    def setSampleSize(self, sampleSize):
        """
        :param sampleSize: The number of examples to randomly sample for each tree.
        :type sampleSize: :class:`int`
        """
        Parameter.checkFloat(sampleSize, 0.0, 1.0)
        self.sampleSize = sampleSize

    def getSampleSize(self):
        """
        :return: The number of examples to randomly sample for each tree.
        """
        return self.sampleSize

    def setSampleReplace(self, sampleReplace):
        """
        :param sampleReplace: A boolean to decide whether to sample with replacement. 
        :type sampleReplace: :class:`bool`
        """
        Parameter.checkBoolean(sampleReplace)
        self.sampleReplace = sampleReplace

    def getSampleReplace(self):
        """
        :return: A boolean to decide whether to sample with replacement. 
        """
        return self.sampleReplace 

    def learnModel(self, X, y):
        """
        Learn a model for a set of examples given as the rows of the matrix X,
        with corresponding labels given in the elements of 1D array y.

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`

        :param y: A vector of labels
        :type y: :class:`ndarray`
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(y, numpy.ndarray)
        Parameter.checkArray(X)
        Parameter.checkArray(y)
        if numpy.unique(y).shape[0] != 2:
            raise ValueError("Can only accept binary labelled data")

        forestList = []
        numSampledExamples = numpy.round(self.sampleSize*X.shape[0])

        for i in range(self.numTrees):
            Util.printIteration(i, 1, self.numTrees, "Tree: ")
            if self.sampleReplace:
                inds = numpy.random.randint(0, X.shape[0], numSampledExamples)
            else:
                inds = numpy.random.permutation(X.shape[0])[0:numSampledExamples]

            treeRank = TreeRank(self.generateLeafRank)
            treeRank.setMaxDepth(self.maxDepth)
            treeRank.setMinSplit(self.minSplit)
            treeRank.setFeatureSize(self.featureSize)
            treeRank.setBestResponse(self.bestResponse)
            treeRank.learnModel(X[inds, :], y[inds])
            forestList.append(treeRank)

        self.forestList = forestList

    def predict(self, X):
        """
        Make a prediction for a set of examples given as the rows of the matrix X.
        The set of scores is the mean over all the trees in the forest.

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`

        :return: A vector of scores corresponding to each example.
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkArray(X)

        scores = numpy.zeros(X.shape[0])

        for i in range(self.numTrees):
            scores += self.forestList[i].predict(X)

        scores = scores/self.numTrees
        return scores

    def getForest(self):
        """
        :return: A list of the trees in the forest.
        """
        return self.forestList

    def getNumTrees(self):
        """
        :return: The number of trees in the forest. 
        """
        return self.numTrees

    def setNumTrees(self, numTrees):
        """
        :param numTrees: The number of trees to generate in the forest.
        :type numTrees: :class:`int`
        """
        Parameter.checkInt(numTrees, 1, float('inf'))
        self.numTrees = numTrees

    def setGenerateLeafRank(self, generateLeafRank):
        """
        :param numTrees: The function that generates leaf rank objects. 
        """
        self.generateLeafRank = generateLeafRank

    def __str__(self):
        outputStr = "TreeRankForest:" + " numTrees=" + str(self.numTrees)
        outputStr += " sampleSize=" + str(self.sampleSize) + " sampleReplace=" + str(self.sampleReplace)
        outputStr += " featureSize=" + str(self.featureSize) + " bestResponse=" + str(self.bestResponse)
        outputStr += " minSplit= " + str(self.minSplit) + " maxDepth=" + str(self.maxDepth)
        return outputStr 