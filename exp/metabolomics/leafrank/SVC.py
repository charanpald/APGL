

import numpy
import logging
import sklearn.svm as svm
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from apgl.metabolomics.leafrank.AbstractWeightedPredictor import AbstractWeightedPredictor

class SVC(AbstractWeightedPredictor):
    """
    A Support Vector Classifier.
    """
    def __init__(self):
        super(SVC, self).__init__()
        self.C = 10.0
        self.kernel = "rbf"
        self.degree = 2
        self.gamma = 1.0
        self.learner = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree)

    def setC(self, C):
        Parameter.checkFloat(C, 0.0, float('inf'))
        self.C = C
        self.learner = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree)

    def getC(self):
        return self.C

    def setGamma(self, gamma):
        Parameter.checkFloat(gamma, 0.0, float('inf'))
        self.gamma = gamma
        self.learner = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree)

    def getGamma(self):
        return self.gamma

    def setDegree(self, degree):
        Parameter.checkFloat(degree, 0.0, float('inf'))
        self.degree = degree
        self.learner = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree)

    def getDegree(self):
        return self.degree

    def setKernel(self, kernel):
        self.kernel = kernel
        self.learner = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree)

    def learnModel(self, X, y):
        if numpy.unique(y).shape[0] != 2:
            raise ValueError("Can only operate on binary data")

        classes = numpy.unique(y)
        worstResponse = classes[classes!=self.bestResponse][0]
        weightDict = {self.bestResponse: self.weight, worstResponse: 1-self.weight}

        self.learner.fit(X, y, class_weight=weightDict)

    def getLearner(self):
        return self.learner

    def predict(self, X):
        return self.learner.predict(X)

    @staticmethod
    def generate(C=10.0, kernel="rbf", gamma=1.0, degree=2.0):
        def generatorFunc():
            svc = SVC()
            svc.setC(C)
            svc.setGamma(gamma)
            svc.setDegree(degree)
            svc.setKernel(kernel)
            return svc
        return generatorFunc

    def __str__(self):
        return str(self.learner)

class SvcGS(AbstractWeightedPredictor):
    def __init__(self):
        self.SVC = SVC()

    def learnModel(self, X, y, folds=3):
        """
        Train using the given examples and labels, however first conduct grid
        search in conjunction with cross validation to find the best parameters.
        """
        #Hard coding this is bad
        Cs = 2**numpy.arange(-2, 6, dtype=numpy.float)
        gammas = 2**numpy.arange(-4, 1, dtype=numpy.float)
        meanAUCs = numpy.zeros((Cs.shape[0], gammas.shape[0]))
        stdAUCs = numpy.zeros((Cs.shape[0], gammas.shape[0]))

        for i in range(Cs.shape[0]):
            for j in range(gammas.shape[0]):
                self.SVC.setC(Cs[i])
                self.SVC.setGamma(gammas[j])
                meanAUCs[i, j], stdAUCs[i, j] = self.SVC.evaluateStratifiedCv(X, y, folds, metricMethod=Evaluator.auc)

        (bestI, bestJ) = numpy.unravel_index(numpy.argmax(meanAUCs), meanAUCs.shape)
        self.SVC.setC(Cs[bestI])
        self.SVC.setGamma(gammas[bestJ])
        logging.debug("AUCs: " + str(meanAUCs))
        logging.debug("Best learner found: " + str(self.SVC))
        self.SVC.learnModel(X, y)

    def predict(self, y):
        return self.SVC.predict(y)

    @staticmethod
    def generate():
        """
        Generate a classifier which does a grid search.
        """
        def generatorFunc():
            svcGS = SvcGS()
            return svcGS
        return generatorFunc

    def setWeight(self, weight):
        self.SVC.setWeight(weight)