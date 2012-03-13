
"""
A functional version of the linear SVM leaf rank.
"""

import numpy
import logging
import sklearn.svm as svm
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from apgl.metabolomics.leafrank.AbstractFunctionalPredictor import AbstractFunctionalPredictor
from apgl.metabolomics.leafrank.LinearSVM import LinearSVM
from apgl.data.Standardiser import Standardiser

class LinearSvmFGs(AbstractFunctionalPredictor):
    def __init__(self):
        super(LinearSvmFGs, self).__init__()
        self.linearSVM = LinearSVM()

    def learnModel(self, X, y, folds=3):
        """
        Train using the given examples and labels, however first conduct grid
        search in conjunction with cross validation to find the best parameters.
        We also conduct filtering with a variety of values. 
        """
        #Hard coding this is bad
        Cs = 2**numpy.arange(-2, 7, dtype=numpy.float)
        #Cs = numpy.array([0.1, 2.0])

        if self.waveletInds == None:
            self.waveletInds = numpy.arange(X.shape[1])

        nonWaveletInds = numpy.setdiff1d(numpy.arange(X.shape[1]),  self.waveletInds)

        Xw = X[:, self.waveletInds]
        Xo = X[:, nonWaveletInds]

        featureInds = numpy.flipud(numpy.argsort(numpy.sum(Xw**2, 0)))
        meanAUCs = numpy.zeros((Cs.shape[0], self.candidatesN.shape[0]))
        stdAUCs = numpy.zeros((Cs.shape[0], self.candidatesN.shape[0]))

        #Standardise the data
        Xw = Standardiser().standardiseArray(Xw)
        Xo = Standardiser().standardiseArray(Xo)

        for i in range(Cs.shape[0]):
            for j in range(self.candidatesN.shape[0]):
                self.linearSVM.setC(Cs[i])
                newX = numpy.c_[Xw[:, featureInds[0:self.candidatesN[j]]], Xo]
                meanAUCs[i, j], stdAUCs[i, j] = self.linearSVM.evaluateStratifiedCv(newX, y, folds, metricMethod=Evaluator.auc)

        (bestI, bestJ) = numpy.unravel_index(numpy.argmax(meanAUCs), meanAUCs.shape)
        self.linearSVM.setC(Cs[bestI])
        self.featureInds = numpy.r_[self.waveletInds[featureInds[0:self.candidatesN[bestJ]]], nonWaveletInds]
        logging.debug("Best learner found: " + str(self.linearSVM) + " N:" + str(self.candidatesN[bestJ]))

        self.standardiser = Standardiser()
        newX = self.standardiser.standardiseArray(X[:, self.featureInds])
        self.linearSVM.learnModel(newX, y)
        
    def predict(self, X):
        newX = self.standardiser.standardiseArray(X[:, self.featureInds])
        return self.linearSVM.predict(newX)

    @staticmethod
    def generate(waveletInds=None):
        """
        Generate a classifier which does a grid search.
        """
        def generatorFunc():
            linearSvm = LinearSvmFGs()
            linearSvm.setWaveletInds(waveletInds)
            return linearSvm
        return generatorFunc

    def setWeight(self, weight):
        self.linearSVM.setWeight(weight)