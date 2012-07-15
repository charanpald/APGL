
import unittest
import numpy
import logging
import sys
from apgl.metabolomics.leafrank.LinearSvmPca import LinearSvmPca

class LinearSvmPcaTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.random.seed(21)
        numFeatures = 200
        numExamples = 200

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))

    def testGenerate(self):
        generate = LinearSvmPca.generate()

        self.X[:, 15:25] = self.X[:, 15:25]*100

        svc = generate()
        svc.learnModel(self.X, self.y)

        predY = svc.predict(self.X)

    def testSetWeight(self):
        learner = LinearSvmPca()
        learner.setWeight(0.8)

if __name__ == '__main__':
    unittest.main()

