
import unittest
import numpy
import logging
import sys
from apgl.metabolomics.leafrank.DecisionTreeF import DecisionTreeF

class DecisionTreeFTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.random.seed(21)
        numFeatures = 200
        numExamples = 200

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))

    def testGenerate(self):
        generate = DecisionTreeF.generate()

        self.X[:, 15:25] = self.X[:, 15:25]*100

        decisionTree = generate()
        decisionTree.setWaveletInds(numpy.arange(100))
        decisionTree.learnModel(self.X, self.y)
        self.assertEquals(numpy.intersect1d(numpy.arange(15,25), decisionTree.getFeatureInds()).shape[0], 10)

        predY = decisionTree.predict(self.X)

        #Now test when all features are wavelets
        decisionTree = generate()
        decisionTree.learnModel(self.X, self.y)
        self.assertEquals(numpy.intersect1d(numpy.arange(15,25), decisionTree.getFeatureInds()).shape[0], 10)

        predY = decisionTree.predict(self.X)


    def testSetWeight(self):
        learner = DecisionTreeF()

        learner.setWeight(0.8)


if __name__ == '__main__':
    unittest.main()

