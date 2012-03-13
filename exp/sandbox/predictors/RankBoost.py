import os 
import subprocess
import tempfile
import numpy
import logging 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
import scikits.learn.cross_val as cross_val
"""
A wrapper for the RankBoost code written in RankLib. Note that RankLib must
be on the current path.
"""

class RankBoost(AbstractPredictor):
    def __init__(self):
        self.iterations = 100
        self.learners = 20
        self.bestResponse = 1

        #This is terrible
        self.libPath = "/home/charanpal/Documents/Postdoc/Code/semisup_rankboost/"
        if not os.path.isdir(self.libPath):
            self.libPath = "/tsi/thane/dhanjal/semisup_rankboost/"

    def setIterations(self, iterations):
        Parameter.checkInt(iterations, 0, float('inf'))
        self.iterations = iterations

    def setLearners(self, learners):
        Parameter.checkInt(learners, 0, float('inf'))
        self.learners = learners

    def setLibPath(self, libPath):
        self.libPath = libPath 

    def saveExamples(self, X, y, fileName):
        file = open(fileName, "w")
        for i in range(X.shape[0]):
            fStr = str(y[i]) + " "
            for j in range(X.shape[1]):
                if j!=X.shape[1]-1:
                    fStr += str(j+1) + ":" + str(X[i, j]) + " "
                else:
                    fStr += str(j+1) + ":" + str(X[i, j])
            fStr += "\n"
            file.write(fStr)
        file.close()

    def getOutputStr(self):
        """
        Return the command line output from the last command
        """
        return self.outputStr 

    def learnModel(self, X, y):
        #Must make sure examples are +1/-1
        newY = numpy.array(y==self.bestResponse, numpy.int)*2 - 1

        trainFileName = tempfile.gettempdir() + "/trainData" + str(numpy.random.randint(numpy.iinfo(numpy.int).max))
        self.modelFileName = tempfile.gettempdir() + "/model" + str(numpy.random.randint(numpy.iinfo(numpy.int).max))
        self.saveExamples(X, newY, trainFileName)
        
        callList = [self.libPath + "ssrankboost-learn", "-t", str(self.iterations), "-n", str(self.learners)]
        callList.extend([trainFileName, self.modelFileName])

        try:
            self.outputStr =  subprocess.check_output(callList)
        except AttributeError:
            subprocess.call(callList)

        modelFile = open(self.modelFileName, "r")
        self.model = modelFile.read()
        modelFile.close()
        os.remove(self.modelFileName)

        os.remove(trainFileName)

    def predict(self, X):
        testFileName = tempfile.gettempdir() + "/testData" + str(numpy.random.randint(numpy.iinfo(numpy.int).max))
        scoreFileName = tempfile.gettempdir() + "/scoreFileName" + str(numpy.random.randint(numpy.iinfo(numpy.int).max))
        self.saveExamples(X, numpy.ones(X.shape[0]), testFileName)

        modelFile = open(self.modelFileName, "w")
        modelFile.write(self.model)
        modelFile.close()
        
        callList = [self.libPath + "ssrankboost-test", testFileName, self.modelFileName, scoreFileName]
        try:
            self.outputStr =  subprocess.check_output(callList)
        except AttributeError:
            subprocess.call(callList)
        os.remove(testFileName)
        os.remove(self.modelFileName)

        #Now read the scores files
        scores = numpy.fromfile(scoreFileName, sep=" ")
        os.remove(scoreFileName)

        return scores

    def modelSelect(self, X, y, folds=5):
        """
        Do model selection for a dataset and then learn using the best parameters
        according to the AUC.
        """
        learnerList = numpy.arange(10, 51, 10)
        meanAUCs = numpy.zeros(learnerList.shape[0])
        stdAUCs = numpy.zeros(learnerList.shape[0])

        for i in range(learnerList.shape[0]):
            self.setLearners(learnerList[i])
            meanAUCs[i], stdAUCs[i] = self.evaluateStratifiedCv(X, y, folds, metricMethod=Evaluator.auc)

        self.setLearners(learnerList[numpy.argmax(meanAUCs)])
        logging.debug("Best learner found: " + str(self))
        self.learnModel(X, y)

    def evaluateCvOuter(self, X, y, folds):
        """
        Computer the average AUC using k-fold cross validation and the linear kernel.
        """
        Parameter.checkInt(folds, 2, float('inf'))
        idx = cross_val.StratifiedKFold(y, folds)
        metricMethods = [Evaluator.auc2, Evaluator.roc]
        trainMetrics, testMetrics = AbstractPredictor.evaluateLearn2(X, y, idx, self.modelSelect, self.predict, metricMethods)

        bestTrainAUCs = trainMetrics[0]
        bestTrainROCs = trainMetrics[1]
        bestTestAUCs = testMetrics[0]
        bestTestROCs = testMetrics[1]

        bestParams = {}
        bestMetaDicts = {}
        allMetrics = [bestTrainAUCs, bestTrainROCs, bestTestAUCs, bestTestROCs]

        return (bestParams, allMetrics, bestMetaDicts)

    def __str__(self):
        outputStr = "RankBoost: learners=" + str(self.learners) + " iterations=" + str(self.iterations)
        return outputStr 