
import numpy
import unittest
import logging
import sys
import rpy2.robjects as robjects
import scikits.learn as learn
from rpy2.robjects.packages import importr
from exp.metabolomics.TreeRankForestR import TreeRankForestR
from apgl.util.Evaluator import Evaluator

class  TreeRankForestRTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        self.baseLib = importr('base')
        self.treeRankLib = importr('TreeRank')
        utilLib = importr("utils")
        utilLib.data("TRdata")

        XY = numpy.array(robjects.r["Gauss2D.learn"]).T
        self.X = XY[:, 0:2]
        self.Y = numpy.array([XY[:, 2]], numpy.int).T

        trainExamples = 500
        self.trainX = self.X[0:trainExamples, :]
        self.trainY = self.Y[0:trainExamples, :]
        self.testX = self.X[trainExamples:, :]
        self.testY = self.Y[trainExamples:, :]

        self.treeRankForest = TreeRankForestR()

    def testInit(self):
        self.treeRankForest = TreeRankForestR()

    #def testLearnModelDataFrame(self):
        #self.treeRankForest.learnModelDataFrame(self.formula, self.XY)

    def testPredictROC(self):
        self.treeRankForest.learnModel(self.trainX, self.trainY)

        roc = self.treeRankForest.predictROC(self.trainX, self.trainY)
        self.assertEquals(roc.shape[1], 2)

        #Compare predictROC to predict + the scikits roc function
        scores = self.treeRankForest.predict(self.trainX)
        fpr, tpr, threshold = learn.metrics.roc_curve(self.trainY.ravel(), scores)
        roc2 = numpy.c_[fpr, tpr]
        roc2 = numpy.r_[numpy.zeros((1, 2)), roc2]

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(roc - roc2) < tol)

        #Now try on test set
        roc = self.treeRankForest.predictROC(self.testX, self.testY)

        scores = self.treeRankForest.predict(self.testX)
        fpr, tpr, threshold = learn.metrics.roc_curve(self.testY.ravel(), scores)
        roc2 = numpy.c_[fpr, tpr]
        roc2 = numpy.r_[numpy.zeros((1, 2)), roc2]

        self.assertTrue(numpy.linalg.norm(roc - roc2) < tol)

    def testPredictScores(self):
        self.treeRankForest.learnModel(self.X, self.Y)

        scores = self.treeRankForest.predictScores(self.X)
        self.assertEquals(scores.shape[0], self.X.shape[0])

    def testAucFromRoc(self):
        self.treeRankForest = TreeRankForestR()
        self.treeRankForest.setLeafRank(self.treeRankForest.getTreeRankLib().LRsvm)
        self.treeRankForest.learnModel(self.X, self.Y)

        roc = self.treeRankForest.predictROC(self.X, self.Y)
        auc = self.treeRankForest.aucFromROC(roc)

        logging.debug(self.treeRankForest)
        self.assertAlmostEquals(auc, 0.7603246, 1)

        self.assertTrue(0 <= auc <= 1)

    def testLearnModel(self):
        numExamples = 10
        numFeatures = 5
        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.random.rand(numExamples)

        self.treeRankForest.learnModel(X, Y)

    @unittest.skip("Demonstrating skipping")
    def testEvaluateCv(self):
        folds = 3

        mean, var = self.treeRankForest.evaluateCv(self.X, self.Y.ravel(), folds, metricMethod=TreeRankR.auc)

        #Compare versus just train/test splits
        trainExamples = 500
        trainX = self.X[0:trainExamples, :]
        trainY = self.Y[0:trainExamples, :]
        testX = self.X[trainExamples:, :]
        testY = self.Y[trainExamples:, :]

        self.treeRankForest.learnModel(trainX, trainY)
        scores = self.treeRankForest.predict(testX)
        #print(TreeRankR.auc(scores, testY))

        self.assertTrue(0 <= mean <= 1)
        self.assertTrue(0 <= var <= 1)

    def testAuc(self):
        self.treeRankForest.learnModel(self.X, self.Y)
        scores = self.treeRankForest.predictScores(self.X)

        auc1 = Evaluator.auc(scores, self.Y.ravel())
        auc2 = self.treeRankForest.aucFromROC(self.treeRankForest.predictROC(self.X, self.Y))

        self.assertAlmostEquals(auc1, auc2, places=4)

    def testGetLRsvm2(self):
        LRsvm2 = self.treeRankForest.getLrRbfSvm()
        self.treeRankForest.setLeafRank(LRsvm2)

        self.treeRankForest.learnModel(self.X, self.Y)
        Y = self.treeRankForest.predict(self.X)

    def testGetLRCart2(self):
        LRCart2 = self.treeRankForest.getLrCart()
        self.treeRankForest.setLeafRank(LRCart2)

        self.treeRankForest.learnModel(self.X, self.Y)
        Y = self.treeRankForest.predict(self.X)

    def learnModel(self, X, Y):
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkArray(X)
        Parameter.checkArray(Y)

        if numpy.unique(Y).shape[0] < 2:
            raise ValueError("Vector of labels must be binary, currently numpy.unique(Y) = " + str(numpy.unique(Y)))

        #If Y is 1D make it 2D
        if Y.ndim == 1:
            Y = numpy.array([Y]).T

        XY = self._getDataFrame(X, Y)
        formula = robjects.Formula('class ~ .')
        self.learnModelDataFrame(formula, XY)

        if self.printMemStats:
            logging.debug(self.getLsos()())
            logging.debug(ProfileUtils.memDisplay(locals()))

    def testEvaluateCvOuter(self):
        folds = 3
        leafRank = self.treeRankForest.getTreeRankLib().LRsvm

        trainExamples = 200
        trainX = self.X[0:trainExamples, :]
        trainY = self.Y[0:trainExamples, :]

        bestParams, allMetrics, bestMetaDicts = self.treeRankForest.evaluateCvOuter(trainX, trainY.ravel(), folds, leafRank)

        for i in range(folds):
            #logging.debug(allMetrics[2])
            logging.debug(allMetrics[2][i])
            #logging.debug(bestParams[i])

if __name__ == '__main__':
    unittest.main()

