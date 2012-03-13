
import unittest
import numpy
from apgl.predictors.PrimalDualCCARegression import PrimalDualCCARegression
from apgl.kernel import * 

class  PrimalDualCCARegressionTest(unittest.TestCase):
    def setUp(self):
        pass

    def testLearnModel(self):
        numExamples = 10
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        Y = X

        tau = 0.0
        tol = 10-6

        kernel = LinearKernel()

        predictor = PrimalDualCCARegression(kernel, tau, tau)
        A = predictor.learnModel(X, Y)

        self.assertTrue(numpy.linalg.norm(numpy.dot(numpy.dot(X, X.T), A) - Y) < tol)
        self.assertTrue(numpy.linalg.norm(predictor.predict(X) - Y) < tol)

    def testSetTau(self):
        tau = 0.0
        tol = 10-6

        kernel = LinearKernel()

        predictor = PrimalDualCCARegression(kernel, tau, tau)

        self.assertEquals(predictor.getTau1(), tau)

        predictor.setTau1(0.1)
        self.assertEquals(predictor.getTau1(), 0.1)

    def testPredict(self):
        numExamples = 10
        numFeatures = 10
        X = numpy.random.rand(numExamples, numFeatures)
        Y = X

        tau = 0.0
        tol = 10-6

        kernel = LinearKernel()

        predictor = PrimalDualCCARegression(kernel, tau, tau)
        A = predictor.learnModel(X, Y)

        testX = X[0:5, :]
        predY = predictor.predict(testX)
        self.assertTrue(numpy.linalg.norm(predY - Y[0:5, :]) < tol)


    def testPredict2(self):
        #Test predicting on low-rank matrices
        numExamples = 10
        numFeatures = 5
        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.random.rand(numExamples, numFeatures)

        tau = 0.0
        tol = 10-6

        kernel = LinearKernel()
        predictor = PrimalDualCCARegression(kernel, tau, tau)
        A = predictor.learnModel(X, Y)

        predY = predictor.predict(X)

        #logging.debug(predY)
        #logging.debug(Y)

if __name__ == '__main__':
    unittest.main()

