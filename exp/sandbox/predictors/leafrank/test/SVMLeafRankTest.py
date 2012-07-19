import unittest
import numpy
import sys
import logging
from exp.sandbox.predictors.leafrank.SVMLeafRank import SVMLeafRank
from apgl.util.Evaluator import Evaluator 

class SVMLeafRankTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numFeatures = 10
        numExamples = 500

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))*2-1

        self.folds = 5 
        self.paramDict = {} 
        self.paramDict["setC"] = 2**numpy.arange(-5, 5, dtype=numpy.float)  

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testPredict(self):
        generator = SVMLeafRank(self.paramDict, self.folds)
        learner = generator.generateLearner(self.X, self.y)
        
        predY = learner.predict(self.X)
        #Seems to work
        auc = learner.getMetricMethod()(predY, self.y)
        auc2 = Evaluator.auc(predY, self.y)    
        
        self.assertEquals(auc, auc2)
        
    def testSetWeight(self):
        #Try weight = 0 and weight = 1
        generator = SVMLeafRank(self.paramDict, self.folds)
        generator.setWeight(0.0)
        learner = generator.generateLearner(self.X, self.y)

        predY = learner.predict(self.X)
        self.assertTrue((predY == -1*numpy.ones(predY.shape[0])).all())
        
        learner = SVMLeafRank(self.paramDict, self.folds)
        learner.setWeight(1.0)
        learner = learner.generateLearner(self.X, self.y)
        predY = learner.predict(self.X)
        self.assertTrue((predY == numpy.ones(predY.shape[0])).all())

if __name__ == '__main__':
    unittest.main()

