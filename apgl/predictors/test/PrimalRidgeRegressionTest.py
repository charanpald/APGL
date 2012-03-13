
import unittest
import numpy
from apgl.predictors.PrimalRidgeRegression import PrimalRidgeRegression
from apgl.util.Sampling import Sampling
from apgl.util.Evaluator import Evaluator 

class  PrimalRidgeRegressionTest(unittest.TestCase):
    def setUp(self):
        numExamples = 50
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        y = X.dot(c) + numpy.random.rand(numExamples)

        self.numExamples = numExamples 
        self.X = X
        self.y = y


    def testLearnModel(self):
        numExamples = 10
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        Y = X

        lmbda = 0.5
        tol = 10-6 

        predictor = PrimalRidgeRegression(lmbda)
        U = predictor.learnModel(X, Y)

        self.assertTrue(numpy.linalg.norm(numpy.dot(X, U) - Y) < tol)
        self.assertTrue(numpy.linalg.norm(predictor.predict(X) - Y) < tol)

        predictor = PrimalRidgeRegression(0.0)
        U = predictor.learnModel(X, Y)

        self.assertTrue(numpy.linalg.norm(U - numpy.eye(numExamples)) < tol)

        #Test case in which y is a single feature
        y = numpy.random.rand(numExamples, 1)

        predictor = PrimalRidgeRegression(0.0)
        U = predictor.learnModel(X, y)

    def testEvaluateLearners(self):
        #This is a tough function to test
        folds = 5
        numExamples = self.numExamples
        X = self.X
        Y = self.y.ravel()

        indexList = Sampling.crossValidation(folds, numExamples)
        splitFunction = lambda trainX, trainY: Sampling.crossValidation(folds, trainX.shape[0])
        learnerIterator = [PrimalRidgeRegression(0.0), PrimalRidgeRegression(1.0), PrimalRidgeRegression(10.0)]
        metricMethods = [Evaluator.binaryError]

        allMetrics, bestLearners = PrimalRidgeRegression.evaluateLearners(X, Y, indexList, splitFunction, learnerIterator, metricMethods, False)



        #Seems to work 


if __name__ == '__main__':
    unittest.main()

