import unittest
import numpy
import sys
import logging
from apgl.metabolomics.leafrank.LinearSVM import LinearSVM, LinearSvmGS

class LinearSVMTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numFeatures = 10
        numExamples = 500

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testPredict(self):
        linearSVM = LinearSvmGS()
        linearSVM.learnModel(self.X, self.y)

        #Seems to work

    def testGenerateGS(self):
        generate = LinearSvmGS.generate()

        learner = generate()
        learner.learnModel(self.X, self.y)
        learner.predict(self.X)

    def testSetWeight(self):
        #Try weight = 0 and weight = 1
        linearSVM = LinearSvmGS()
        linearSVM.setWeight(0.0)
        linearSVM.learnModel(self.X, self.y)

        predY = linearSVM.predict(self.X)
        self.assertTrue((predY == numpy.zeros(predY.shape[0])).all())

        linearSVM.setWeight(1.0)
        linearSVM.learnModel(self.X, self.y)
        predY = linearSVM.predict(self.X)
        self.assertTrue((predY == numpy.ones(predY.shape[0])).all())

    def testSetWeight2(self):
        #Try weight = 0 and weight = 1
        linearSVM = LinearSVM()
        linearSVM.setWeight(0.0)
        linearSVM.learnModel(self.X, self.y)

        predY = linearSVM.predict(self.X)
        self.assertTrue((predY == numpy.zeros(predY.shape[0])).all())

        linearSVM.setWeight(1.0)
        linearSVM.learnModel(self.X, self.y)
        predY = linearSVM.predict(self.X)
        self.assertTrue((predY == numpy.ones(predY.shape[0])).all())

if __name__ == '__main__':
    unittest.main()

