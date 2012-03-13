import unittest
import numpy
import logging
import sys
from apgl.predictors.KernelRidgeRegression import KernelRidgeRegression
from apgl.predictors.KernelShiftRegression import KernelShiftRegression
from apgl.kernel.LinearKernel import LinearKernel
from apgl.util import *
from apgl.data.Standardiser import Standardiser



class  KernelShiftRegressionTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def testLearnModel(self):
        numExamples = 50
        numFeatures = 200
        preprocessor = Standardiser()
        X = numpy.random.randn(numExamples, numFeatures)
        X = preprocessor.standardiseArray(X)
        c = numpy.random.rand(numFeatures)
        y = numpy.dot(X, c)

        tol = 0.05
        kernel = LinearKernel()
        lmbda = 0.0001
        predictor = KernelShiftRegression(kernel, lmbda)

        alpha, b = predictor.learnModel(X, y)
        predY = predictor.predict(X)

        self.assertTrue(Evaluator.rootMeanSqError(y, predY) < tol)

        #Try increasing y
        y = y + 5
        lmbda = 0.2
        predictor = KernelShiftRegression(kernel, lmbda)
        alpha, b = predictor.learnModel(X, y)
        predY = predictor.predict(X)

        self.assertTrue(numpy.abs(b - 5) < 0.1)
        self.assertTrue(Evaluator.rootMeanSqError(y, predY) < 0.1)

        #Try making prediction for multilabel Y
        C = numpy.random.rand(numFeatures, numFeatures)
        Y = numpy.dot(X, C)

        predictor = KernelShiftRegression(kernel, lmbda)
        alpha, b = predictor.learnModel(X, Y)
        predY = predictor.predict(X)

        self.assertTrue(Evaluator.rootMeanSqError(Y, predY) < 0.1)

        #Now, shift the data 
        s = numpy.random.rand(numFeatures)
        Y = Y + s

        predictor = KernelShiftRegression(kernel, lmbda)
        alpha, b = predictor.learnModel(X, Y)
        predY = predictor.predict(X)

        self.assertTrue(numpy.linalg.norm(b - s) < 0.1)
        self.assertTrue(Evaluator.rootMeanSqError(Y, predY) < 0.1)

    def testLearnModel2(self):
        numExamples = 200
        numFeatures = 100

        X = numpy.random.randn(numExamples, numFeatures)
        y = numpy.random.randn(numExamples)

        preprocessor = Standardiser()
        X = preprocessor.standardiseArray(X)

        tol = 10**-3
        kernel = LinearKernel()

        #Try using a low-rank matrix 
        lmbda = 0.001
        predictor = KernelShiftRegression(kernel, lmbda)

        alpha, b = predictor.learnModel(X, y)
        predY = predictor.predict(X)

        logging.debug((numpy.linalg.norm(y)))
        logging.debug((numpy.linalg.norm(predY - y)))

        

if __name__ == '__main__':
    unittest.main()

