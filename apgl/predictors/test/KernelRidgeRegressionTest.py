
import unittest
from apgl.predictors.KernelRidgeRegression import KernelRidgeRegression
from apgl.kernel.LinearKernel import LinearKernel
from apgl.util.Util import Util
from apgl.data.Standardiser import Standardiser
import numpy
import logging
import sys 

class  KernelRidgeRegressionTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    def testLearnModel(self):
        numExamples = 50
        numFeatures = 200

        X = numpy.random.randn(numExamples, numFeatures)
        y = numpy.random.randn(numExamples)

        preprocessor = Standardiser()
        X = preprocessor.standardiseArray(X)

        tol = 10**-3
        kernel = LinearKernel()

        #Compare Linear kernel with linear ridge regression 
        lmbda = 0.1
        predictor = KernelRidgeRegression(kernel, lmbda)

        alpha = predictor.learnModel(X, y)
        predY = predictor.predict(X)

        K = numpy.dot(X, X.T)
        alpha2 = numpy.dot(numpy.linalg.inv(K+lmbda*numpy.eye(numExamples)), y)
        predY2 = X.dot(numpy.linalg.inv(numpy.dot(X.T, X) + lmbda*numpy.eye(numFeatures))).dot(X.T).dot(y)



        #logging.debug(numpy.linalg.norm(alpha - alpha2))

        self.assertTrue(numpy.linalg.norm(alpha - alpha2) < tol)
        self.assertTrue(numpy.linalg.norm(predY - predY2) < tol)

        lmbda = 0.5
        predictor = KernelRidgeRegression(kernel, lmbda)

        alpha = predictor.learnModel(X, y)
        predY = predictor.predict(X)

        K = numpy.dot(X, X.T)
        alpha2 = numpy.dot(numpy.linalg.inv(K+lmbda*numpy.eye(numExamples)), y)
        predY2 = X.dot(numpy.linalg.inv(numpy.dot(X.T, X) + lmbda*numpy.eye(numFeatures))).dot(X.T).dot(y)

        self.assertTrue(numpy.linalg.norm(alpha - alpha2) < tol)
        self.assertTrue(numpy.linalg.norm(predY - predY2) < tol)

        #Now test on an alternative test set
        numTestExamples = 50
        testX = numpy.random.randn(numTestExamples, numFeatures)
        predictor = KernelRidgeRegression(kernel, lmbda)

        alpha = predictor.learnModel(X, y)
        predY = predictor.predict(testX)

        K = numpy.dot(X, X.T)
        alpha2 = numpy.dot(numpy.linalg.inv(K+lmbda*numpy.eye(numExamples)), y)
        predY2 = testX.dot(numpy.linalg.inv(numpy.dot(X.T, X) + lmbda*numpy.eye(numFeatures))).dot(X.T).dot(y)

        self.assertTrue(numpy.linalg.norm(alpha - alpha2) < tol)
        self.assertTrue(numpy.linalg.norm(predY - predY2) < tol)

        #Use the method against a multi-label example
        Y = numpy.random.randn(numExamples, numFeatures)

        alpha = predictor.learnModel(X, Y)

        self.assertTrue(alpha.shape == (numExamples, numFeatures))


    def testClassify(self):
        numExamples = 10
        numFeatures = 20

        X = numpy.random.randn(numExamples, numFeatures)
        y = numpy.sign(numpy.random.randn(numExamples))
        logging.debug(y)

        preprocessor = Standardiser()
        X = preprocessor.standardiseArray(X)

        tol = 10**-5
        lmbda = 1.0
        kernel = LinearKernel()

        predictor = KernelRidgeRegression(kernel, lmbda)
        predictor.learnModel(X, y)
        classY, predY = predictor.classify(X)

        self.assertTrue(numpy.logical_or(classY == 1, classY == -1).all() ) 

if __name__ == '__main__':
    unittest.main()

