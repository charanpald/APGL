
"""
A functional version of the Decision Tree Leaf Rank. 
"""

import numpy
import logging
from apgl.util.Evaluator import Evaluator
from exp.metabolomics.leafrank.DecisionTree import DecisionTree
from exp.metabolomics.leafrank.AbstractFunctionalPredictor import AbstractFunctionalPredictor
from apgl.data.Standardiser import Standardiser

class DecisionTreeF(AbstractFunctionalPredictor):
    def __init__(self):
        super(DecisionTreeF, self).__init__()
        self.decisionTree = DecisionTree()

    def learnModel(self, X, y, folds=3):
        """
        Train using the given examples and labels, however first conduct grid
        search in conjunction with cross validation to find the best parameters.
        We also conduct filtering with a variety of values.
        """
        if self.waveletInds == None:
            self.waveletInds = numpy.arange(X.shape[1])

        nonWaveletInds = numpy.setdiff1d(numpy.arange(X.shape[1]),  self.waveletInds)

        Xw = X[:, self.waveletInds]
        Xo = X[:, nonWaveletInds]

        featureInds = numpy.flipud(numpy.argsort(numpy.sum(Xw**2, 0)))
        meanAUCs = numpy.zeros(self.candidatesN.shape[0])
        stdAUCs = numpy.zeros(self.candidatesN.shape[0])

        #Standardise the data
        Xw = Standardiser().standardiseArray(Xw)
        Xo = Standardiser().standardiseArray(Xo)

        for i in range(self.candidatesN.shape[0]):
            newX = numpy.c_[Xw[:, featureInds[0:self.candidatesN[i]]], Xo]
            meanAUCs[i], stdAUCs[i] = self.decisionTree.evaluateStratifiedCv(newX, y, folds, metricMethod=Evaluator.auc)

        bestI = numpy.argmax(meanAUCs)
        self.featureInds = numpy.r_[self.waveletInds[featureInds[0:self.candidatesN[bestI]]], nonWaveletInds]
        logging.debug("Best learner found: " + str(self.decisionTree) + " N:" + str(self.candidatesN[bestI]))

        self.standardiser = Standardiser()
        newX = self.standardiser.standardiseArray(X[:, self.featureInds])
        self.decisionTree.learnModel(newX, y)

    def predict(self, X):
        newX = self.standardiser.standardiseArray(X[:, self.featureInds])
        return self.decisionTree.predict(newX)

    @staticmethod
    def generate(waveletInds=None):
        """
        Generate a classifier which does a grid search.
        """
        def generatorFunc():
            decisionTree = DecisionTreeF()
            decisionTree.setWaveletInds(waveletInds)
            return decisionTree
        return generatorFunc

    def setWeight(self, weight):
        self.decisionTree.setWeight(weight)