

"""
A functional version of the linear SVM leaf rank.
"""

import numpy
import logging
import sklearn.svm as svm
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from apgl.metabolomics.leafrank.AbstractWeightedPredictor import AbstractWeightedPredictor
from apgl.metabolomics.leafrank.LinearSVM import LinearSVM
from apgl.data.Standardiser import Standardiser
from apgl.features.PrimalPCA import PrimalPCA

class LinearSvmPca(AbstractWeightedPredictor):
    """
    A LeafRank which uses PCA in conjunction with a linear SVM to perform prediction. 
    """
    def __init__(self):
        super(LinearSvmPca, self).__init__()
        self.linearSVM = LinearSVM()
        self.candidatesN = numpy.arange(20, 100, 10)
        
    def learnModel(self, X, y, folds=3):
        """
        Train using the given examples and labels, however first conduct grid
        search in conjunction with cross validation to find the best parameters.
        We also conduct filtering with a variety of values.
        """
        #Hard coding this is bad
        Cs = 2**numpy.arange(-3, 6, dtype=numpy.float)

        meanAUCs = numpy.zeros((Cs.shape[0], self.candidatesN.shape[0]))
        stdAUCs = numpy.zeros((Cs.shape[0], self.candidatesN.shape[0]))

        maxN = numpy.max(self.candidatesN)
        self.pca = PrimalPCA(maxN)
        self.pca.learnModel(X)
        newX = self.pca.project(X)

        for i in range(Cs.shape[0]):
            for j in range(self.candidatesN.shape[0]):
                self.linearSVM.setC(Cs[i])
                meanAUCs[i, j], stdAUCs[i, j] = self.linearSVM.evaluateStratifiedCv(newX[:, 0:j], y, folds, metricMethod=Evaluator.auc)

        print(meanAUCs)

        (bestI, bestJ) = numpy.unravel_index(numpy.argmax(meanAUCs), meanAUCs.shape)
        self.linearSVM.setC(Cs[bestI])
        self.pca.setK(self.candidatesN[bestJ])
        logging.debug("Best learner found: " + str(self.linearSVM) + " N:" + str(self.candidatesN[bestJ]))

        newX = newX[:, 0:self.candidatesN[bestJ]]
        self.linearSVM.learnModel(newX, y)

    def predict(self, X):
        newX = self.pca.project(X)
        return self.linearSVM.predict(newX)

    @staticmethod
    def generate(waveletInds=None):
        """
        Generate a classifier which does a grid search.
        """
        def generatorFunc():
            linearSvm = LinearSvmPca()
            return linearSvm
        return generatorFunc

    def setWeight(self, weight):
        self.linearSVM.setWeight(weight)