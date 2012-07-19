import unittest
import numpy
import logging
import sys
from apgl.metabolomics.leafrank.SVC import SVC, SvcGS

class SvcTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numFeatures = 10
        numExamples = 500

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testGenerate(self):
        generate = SVC.generate()

        svc = generate()

        svc.learnModel(self.X, self.y)
        predY = svc.predict(self.X)

        #print(predY)

    def testGenerateRBFGS(self):
        generate = SvcGS.generate()

        learner = generate()
        learner.learnModel(self.X, self.y)
        learner.predict(self.X)
        
    def testSetWeight(self):
        #Try weight = 0 and weight = 1
        svc = SVC()
        svc.setWeight(0.0)
        svc.learnModel(self.X, self.y)

        predY = svc.predict(self.X)
        self.assertTrue((predY == numpy.zeros(predY.shape[0])).all())

        svc.setWeight(1.0)
        svc.learnModel(self.X, self.y)
        predY = svc.predict(self.X)
        self.assertTrue((predY == numpy.ones(predY.shape[0])).all())

    def testSetWeight2(self):
        #Try weight = 0 and weight = 1
        svc = SvcGS()
        svc.setWeight(0.0)
        svc.learnModel(self.X, self.y)

        predY = svc.predict(self.X)
        self.assertTrue((predY == numpy.zeros(predY.shape[0])).all())

        svc.setWeight(1.0)
        svc.learnModel(self.X, self.y)
        predY = svc.predict(self.X)
        self.assertTrue((predY == numpy.ones(predY.shape[0])).all())

if __name__ == '__main__':
    unittest.main()

