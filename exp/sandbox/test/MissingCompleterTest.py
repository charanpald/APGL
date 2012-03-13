
from apgl.features.MissingCompleter import MissingCompleter
import numpy
import unittest
import logging

class MissingCompleterTest(unittest.TestCase):
    def setUp(self):
        pass


    def testLearnModel(self):
        numExamples = 50
        numPartialFeatures = 7
        numFeatures = 10 

        X = numpy.zeros((numExamples, numFeatures))
        X[:, 0:numPartialFeatures] = numpy.random.rand(numExamples, numPartialFeatures)
        C = numpy.random.rand(numPartialFeatures, numFeatures-numPartialFeatures)

        X[:, numPartialFeatures:numFeatures] = numpy.dot(X[:, 0:numPartialFeatures], C)

        K = numpy.dot(X, X.T)
        KHat = numpy.dot(X[:, 0:numPartialFeatures], X[:, 0:numPartialFeatures].T)

        missingCompleter = MissingCompleter()
        k = 7
        
        alpha, beta = missingCompleter.learnModel(KHat, K, k)

        #Check the scaling of alpha and beta
        #Check alpha is orthogonal

        KHatSq = numpy.dot(KHat, KHat)

        logging.debug((numpy.dot(numpy.dot(alpha.T, KHatSq), alpha)))
        logging.debug((numpy.dot(numpy.dot(beta.T, KHat), beta)))

        logging.debug(alpha)
        logging.debug(beta)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()