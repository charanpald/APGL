"""
A wrapper for rank svm code.
"""
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.util.Parameter import Parameter
import scikits.learn.cross_val as cross_val
from apgl.util.Evaluator import Evaluator
import svmlight
import numpy
import logging

class RankSVM(AbstractPredictor):
    """
    Learn the RankSVM algorithm. All QIds are the same. 
    """
    def __init__(self):
        self.C = 1
        self.gamma = 1.0
        self.kernel = "linear"

    def setC(self, C):
        """
        Penalty param for SVM 
        """
        Parameter.checkFloat(C, 0.0, float('inf'))
        self.C = C

    def setGamma(self, gamma):
        Parameter.checkFloat(gamma, 0.0, float('inf'))
        self.gamma = gamma

    def setKernel(self, kernel):
        """
        Can be 'linear', 'polynomial', 'rbf' and 'sigmoid'
        """
        self.kernel = kernel 

    def __createData(self, X, y=None):
        """
        Create a dataset in the format of svm light.
        """
        dataList = []

        if y==None:
            y = numpy.ones(X.shape[0])

        for i in range(X.shape[0]):
            featureList = []
            for j in range(X.shape[1]):
                featureList.append((j+1, X[i, j]))
            dataList.append((y[i], featureList))

        return dataList

    def learnModel(self, X, y):
        dataList = self.__createData(X, y)
        self.model = svmlight.learn(dataList, type='ranking', verbosity=0, kernel=self.kernel, C=self.C, gamma=self.gamma)

    def predict(self, X):
        dataList = self.__createData(X)
        return numpy.array(svmlight.classify(self.model, dataList))

    def modelSelectLinear(self, X, y, folds=5):
        """
        Do model selection for a dataset and then learn using the best parameters
        according to the AUC. 
        """
        if self.kernel != "linear":
            raise ValueError("Must use linear kernel")

        Cs = 2**numpy.arange(-3, 3, dtype=numpy.float)
        meanAUCs = numpy.zeros(Cs.shape[0])
        stdAUCs = numpy.zeros(Cs.shape[0])

        for i in range(Cs.shape[0]):
            self.setC(Cs[i])
            meanAUCs[i], stdAUCs[i] = self.evaluateStratifiedCv(X, y, folds, metricMethod=Evaluator.auc)

        self.setC(Cs[numpy.argmax(meanAUCs)])
        logging.debug("Best learner found: " + str(self))
        self.learnModel(X, y)

    def modelSelectRBF(self, X, y, folds=3):
        """
        Do model selection for a dataset and then learn using the best parameters
        according to the AUC.
        """
        if self.kernel != "rbf":
            raise ValueError("Must use rbf kernel")

        Cs = 2**numpy.arange(-3, 3, dtype=numpy.float)
        gammas = 2**numpy.arange(-4, -1, dtype=numpy.float)
        meanAUCs = numpy.zeros((Cs.shape[0], gammas.shape[0]))
        stdAUCs = numpy.zeros((Cs.shape[0], gammas.shape[0]))

        for i in range(Cs.shape[0]):
            self.setC(Cs[i])
            for j in range(gammas.shape[0]):
                self.setGamma(gammas[j])
                meanAUCs[i, j], stdAUCs[i, j] = self.evaluateStratifiedCv(X, y, folds, metricMethod=Evaluator.auc)

        (bestI, bestJ) = numpy.unravel_index(numpy.argmax(meanAUCs), meanAUCs.shape)

        self.setC(Cs[bestI])
        self.setGamma(gammas[bestJ])
        logging.debug("Best learner found: " + str(self))
        self.learnModel(X, y)

    def evaluateCvOuter(self, X, y, folds):
        """
        Computer the average AUC using k-fold cross validation and the linear kernel. 
        """
        Parameter.checkInt(folds, 2, float('inf'))
        idx = cross_val.StratifiedKFold(y, folds)
        metricMethods = [Evaluator.auc2, Evaluator.roc]

        if self.kernel == "linear":
            logging.debug("Running linear rank SVM ")
            trainMetrics, testMetrics = AbstractPredictor.evaluateLearn2(X, y, idx, self.modelSelectLinear, self.predict, metricMethods)
        elif self.kernel == "rbf":
            logging.debug("Running RBF rank SVM")
            trainMetrics, testMetrics = AbstractPredictor.evaluateLearn2(X, y, idx, self.modelSelectRBF, self.predict, metricMethods)

        bestTrainAUCs = trainMetrics[0]
        bestTrainROCs = trainMetrics[1]
        bestTestAUCs = testMetrics[0]
        bestTestROCs = testMetrics[1]

        bestParams = {}
        bestMetaDicts = {}
        allMetrics = [bestTrainAUCs, bestTrainROCs, bestTestAUCs, bestTestROCs]

        return (bestParams, allMetrics, bestMetaDicts)
        

    def __str__(self):
        outputStr = "RankSVM: C=" + str(self.C) + " kernel=" + str(self.kernel)
        outputStr += " gamma= " + str(self.gamma)
        return outputStr 
