
import unittest
import numpy
import logging
from exp.sandbox.predictors.PrimalWeightedRidgeRegression import PrimalWeightedRidgeRegression


class  PrimalWeightedRidgeRegressionTest(unittest.TestCase):
    def setUp(self):
        pass

    def testLearnModel(self):
        numExamples = 10
        numFeatures = 1

        X = numpy.random.rand(numExamples, numFeatures)
        y = X.ravel()

        lmbda = 0.5
        tol = 10-6

        predictor = PrimalWeightedRidgeRegression(lmbda)
        u = predictor.learnModel(X, y)

        self.assertTrue(numpy.linalg.norm(numpy.dot(X, u) - y) < tol)
        self.assertTrue(numpy.linalg.norm(predictor.predict(X) - y) < tol)

        predictor = PrimalWeightedRidgeRegression(0.0)
        u = predictor.learnModel(X, y)

        self.assertTrue(numpy.linalg.norm(u - numpy.ones(numExamples)) < tol)

        #Now we need to test a case in which the examples are weighted badly
        numExamples = 20 
        numFeatures = 5
        X = numpy.random.rand(numExamples, numFeatures)
        y = numpy.zeros(numExamples)

        y[5] = 1
        y[6] = 1

        lmbda = 0.5
        alpha = 1.0
        predictor = PrimalWeightedRidgeRegression(lmbda, alpha)

        predictor.learnModel(X, y)
        predY = predictor.predict(X)

        errors = numpy.abs(predY - y)
        logging.debug(errors)
        logging.debug((numpy.linalg.norm(predY - y)))

        alpha = 0.0
        predictor = PrimalWeightedRidgeRegression(lmbda, alpha)
        predictor.learnModel(X, y)
        predY = predictor.predict(X)
        errors2 = numpy.abs(predY - y)
        logging.debug(errors2)
        logging.debug((numpy.linalg.norm(predY - y)))

        #We expect errors on 5,6 to be lower 
        self.assertTrue(errors[5] < errors2[5])
        self.assertTrue(errors[6] < errors2[6])
        

if __name__ == '__main__':
    unittest.main()

