import unittest
import numpy
from apgl.kernel.LinearKernel import LinearKernel


class LinearKernelTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testEvaluate(self):
        numExamples = 10
        numFeatures = 5

        X = numpy.random.rand(numExamples, numFeatures)
        K1 = numpy.dot(X, X.T)

        linearKernel = LinearKernel()
        K2 = linearKernel.evaluate(X, X)

        self.assertTrue(numpy.linalg.norm(K1- K2) <= 0.01)

        numExamples2 = 5
        X2 = numpy.random.rand(numExamples2, numFeatures)
        K1 = numpy.dot(X, X2.T)
        K2 = linearKernel.evaluate(X, X2)

        self.assertTrue(numpy.linalg.norm(K1- K2) <= 0.01)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()


