import numpy
import logging
import sklearn.svm as svm
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from apgl.metabolomics.leafrank.AbstractWeightedPredictor import AbstractWeightedPredictor

class LinearSVM(AbstractWeightedPredictor):
    def __init__(self):
        super(LinearSVM, self).__init__()
        self.C = 10
        self.learner = svm.LinearSVC(C=self.C)

    def setC(self, C):
        Parameter.checkFloat(C, 0.0, float('inf'))
        self.learner = svm.LinearSVC(C=C)
        self.C = C

    def getC(self):
        return self.C

    def learnModel(self, X, y):
        """
        Train using the given examples and labels.
        """
        if numpy.unique(y).shape[0] != 2:
            print(y)
            raise ValueError("Can only operate on binary data")

        classes = numpy.unique(y)
        worstResponse = classes[classes!=self.bestResponse][0]
        self.learner.fit(X, y, class_weight={self.bestResponse: self.weight, worstResponse: 1-self.weight})

    def getLearner(self):
        return self.learner

    def predict(self, X):
        return self.learner.predict(X)

    @staticmethod
    def generate(C=10.0):
        """
        Generate a classifier with a fixed parameter set.
        """
        def generatorFunc():
            linearSvm = LinearSVM()
            linearSvm.setC(C)
            return linearSvm
        return generatorFunc

    def __str__(self):
        return str(self.learner)

class LinearSvmGS(AbstractWeightedPredictor):
    def __init__(self):
        self.linearSVM = LinearSVM()

    def learnModel(self, X, y, folds=3):
        """
        Train using the given examples and labels, however first conduct grid
        search in conjunction with cross validation to find the best parameters.
        """
        #Hard coding this is bad
        Cs = 2**numpy.arange(-2, 7, dtype=numpy.float)
        meanAUCs = numpy.zeros(Cs.shape[0])
        stdAUCs = numpy.zeros(Cs.shape[0])

        for i in range(Cs.shape[0]):
            self.linearSVM.setC(Cs[i])
            meanAUCs[i], stdAUCs[i] = self.linearSVM.evaluateStratifiedCv(X, y, folds, metricMethod=Evaluator.auc)

        self.linearSVM.setC(Cs[numpy.argmax(meanAUCs)])
        logging.debug("Best learner found: " + str(self.linearSVM))
        self.linearSVM.learnModel(X, y)

    def predict(self, X):
        return self.linearSVM.predict(X)

    @staticmethod
    def generate():
        """
        Generate a classifier which does a grid search.
        """
        def generatorFunc():
            linearSvm = LinearSvmGS()
            return linearSvm
        return generatorFunc

    def setWeight(self, weight):
        self.linearSVM.setWeight(weight)