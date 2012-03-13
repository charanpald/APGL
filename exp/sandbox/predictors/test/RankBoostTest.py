import numpy
import unittest
import logging
import sys 
from apgl.metabolomics.TreeRank import TreeRank
from apgl.metabolomics.leafrank.LinearSVM import LinearSVM
from apgl.metabolomics.RankBoost import RankBoost
from apgl.util.Evaluator import Evaluator
from apgl.data.Standardiser import Standardiser

class RankBoostTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr(all="raise")

        numExamples = 100
        numFeatures = 10
        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.array(numpy.sign(self.X.dot(c) - numpy.mean(self.X.dot(c))), numpy.int)

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        self.X = Standardiser().standardiseArray(self.X)
        #self.X = numpy.round(self.X, 3)


    def testInit(self):
        rankBoost = RankBoost()

    def testLearnModel(self):
        rankBoost = RankBoost()
        rankBoost.learnModel(self.X, self.y)

    def testPredict(self):
        rankBoost = RankBoost()
        rankBoost.learnModel(self.X, self.y)
        predY = rankBoost.predict(self.X)

        self.assertTrue(Evaluator.auc(predY, self.y) <= 1.0 and Evaluator.auc(predY, self.y) >= 0.0)

    def testSetIterations(self):
        rankBoost = RankBoost()
        rankBoost.setIterations(40)
        rankBoost.learnModel(self.X, self.y)
        predY = rankBoost.predict(self.X)

        #Checked by looking at outputStr


    def testSetLearners(self):
        rankBoost = RankBoost()
        rankBoost.setIterations(10)
        rankBoost.setLearners(20)
        rankBoost.learnModel(self.X, self.y)

        predY = rankBoost.predict(self.X)

    def testEvaluateCvOuter(self):
        folds = 3
        rankBoost = RankBoost()
        (bestParams, allMetrics, bestMetaDicts) = rankBoost.evaluateCvOuter(self.X, self.y, folds)

        self.assertEquals(len(allMetrics[0]), folds)
        self.assertEquals(len(allMetrics[2]), folds)

    def testStr(self):
        rankBoost = RankBoost()
