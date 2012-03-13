
"""
A functional version of the SVM leaf rank.
"""

import numpy
import logging
from apgl.util.Evaluator import Evaluator
from apgl.metabolomics.leafrank.AbstractFunctionalPredictor import AbstractFunctionalPredictor
from apgl.metabolomics.leafrank.SVC import SVC
from apgl.data.Standardiser import Standardiser

class SvcFGs(AbstractFunctionalPredictor):
    def __init__(self):
        super(SvcFGs, self).__init__()
        self.SVC = SVC()

    def learnModel(self, X, y, folds=3):
        """
        Train using the given examples and labels, however first conduct grid
        search in conjunction with cross validation to find the best parameters.
        We also conduct filtering with a variety of values.
        """
        #Hard coding this is bad
        Cs = 2**numpy.arange(-2, 6, dtype=numpy.float)
        gammas = 2**numpy.arange(-5, 0, dtype=numpy.float)

        if self.waveletInds == None:
            self.waveletInds = numpy.arange(X.shape[1])

        nonWaveletInds = numpy.setdiff1d(numpy.arange(X.shape[1]),  self.waveletInds)

        Xw = X[:, self.waveletInds]
        Xo = X[:, nonWaveletInds]

        featureInds = numpy.flipud(numpy.argsort(numpy.sum(Xw**2, 0)))
        meanAUCs = numpy.zeros((Cs.shape[0], gammas.shape[0], self.candidatesN.shape[0]))
        stdAUCs = numpy.zeros((Cs.shape[0], gammas.shape[0], self.candidatesN.shape[0]))

        #Standardise the data
        Xw = Standardiser().standardiseArray(Xw)
        Xo = Standardiser().standardiseArray(Xo)

        for i in range(Cs.shape[0]):
            for j in range(gammas.shape[0]): 
                for k in range(self.candidatesN.shape[0]):
                    self.SVC.setC(Cs[i])
                    self.SVC.setGamma(gammas[j])
                    newX = numpy.c_[Xw[:, featureInds[0:self.candidatesN[k]]], Xo]
                    meanAUCs[i, j, k], stdAUCs[i, j, k] = self.SVC.evaluateStratifiedCv(newX, y, folds, metricMethod=Evaluator.auc)

        (bestI, bestJ, bestK) = numpy.unravel_index(numpy.argmax(meanAUCs), meanAUCs.shape)
        self.SVC.setC(Cs[bestI])
        self.SVC.setGamma(gammas[bestJ])
        self.featureInds = numpy.r_[self.waveletInds[featureInds[0:self.candidatesN[bestK]]], nonWaveletInds]
        logging.debug("Best learner found: " + str(self.SVC) + " N:" + str(self.candidatesN[bestK]))

        self.standardiser = Standardiser()
        newX = self.standardiser.standardiseArray(X[:, self.featureInds])
        self.SVC.learnModel(newX, y)

    def predict(self, X):
        newX = self.standardiser.standardiseArray(X[:, self.featureInds])
        return self.SVC.predict(newX)

    @staticmethod
    def generate(waveletInds=None):
        """
        Generate a classifier which does a grid search.
        """
        def generatorFunc():
            svc = SvcFGs()
            svc.setWaveletInds(waveletInds)
            return svc
        return generatorFunc

    def setWeight(self, weight):
        self.SVC.setWeight(weight)