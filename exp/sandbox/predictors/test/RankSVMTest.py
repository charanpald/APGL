import numpy
import sys
import logging
import unittest
from exp.sandbox.predictors.RankSVM import RankSVM
from apgl.util.Evaluator import Evaluator

class  RankSVMTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr(all="raise")

        numExamples = 50
        numFeatures = 10
        self.X = numpy.random.rand(numExamples, numFeatures)
        self.y = numpy.array(numpy.sign(numpy.random.rand(numExamples)-0.5), numpy.int)

    def testInit(self):
        rankSVM = RankSVM()

    def testLearnModel(self):
        rankSVM = RankSVM()
        rankSVM.learnModel(self.X, self.y)

    def testPredict(self):
        rankSVM = RankSVM()
        rankSVM.learnModel(self.X, self.y)
        predY = rankSVM.predict(self.X)
        #print(Evaluator.auc(predY, self.y))

    def testSetC(self):
        rankSVM = RankSVM()
        rankSVM.setC(100.0)
        rankSVM.learnModel(self.X, self.y)
        predY = rankSVM.predict(self.X)
        auc1 = Evaluator.auc(predY, self.y)

        rankSVM.setC(0.1)
        rankSVM.learnModel(self.X, self.y)
        predY = rankSVM.predict(self.X)
        auc2 = Evaluator.auc(predY, self.y)

        self.assertTrue(auc1 != auc2)

    def testEvaluateCvOuter(self):
        folds = 3 
        rankSVM = RankSVM()
        (bestParams, allMetrics, bestMetaDicts) = rankSVM.evaluateCvOuter(self.X, self.y, folds)

        self.assertEquals(len(allMetrics[0]), folds)
        self.assertEquals(len(allMetrics[2]), folds)

        #for i in allMetrics[1]:
        #    print(i)

        #Now try the RBF version
        rankSVM.setKernel("rbf")
        (bestParams, allMetrics, bestMetaDicts) = rankSVM.evaluateCvOuter(self.X, self.y, folds)

    def testStr(self):
        rankSVM = RankSVM()
        #print(rankSVM)

    
    def testModelSelectRBF(self):
        folds = 3
        rankSVM = RankSVM()
        rankSVM.setKernel("rbf")

        #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        rankSVM.modelSelectRBF(self.X, self.y, folds)

if __name__ == '__main__':
    unittest.main()

