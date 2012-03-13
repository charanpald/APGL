
import unittest
import numpy
from apgl.kernel.PolyKernel import PolyKernel

class  PolyKernelTest(unittest.TestCase):
    def setUp(self):
        pass

    def testEvaluate(self):
        numExamples = 10
        numFeatures = 5

        X = numpy.random.randn(numExamples, numFeatures)

        tol = 10**-6
        b = 1
        degree = 2
        kernel = PolyKernel()

        K = kernel.evaluate(X, X)

        K2 = numpy.zeros((numExamples, numExamples))

        for i in range(numExamples):
            for j in range(numExamples):
                K2[i, j] = (numpy.dot(X[i, :], X[j, :]) + b)**degree

        self.assertTrue(numpy.linalg.norm(K - K2) < tol)
        self.assertTrue(numpy.linalg.norm(K2 - K) < tol)

        #Now try a different set of examples for 2nd arg
        numX2Examples = 20
        X2 = numpy.random.randn(numX2Examples, numFeatures)
        K = kernel.evaluate(X, X2)
        K2 = numpy.zeros((numExamples, numX2Examples))
        K3 = kernel.evaluate(X2, X)

        for i in range(numExamples):
            for j in range(numX2Examples):
                K2[i, j] = (numpy.dot(X[i, :], X2[j, :]) + b)**degree


        self.assertTrue(numpy.linalg.norm(K2 - K) < tol)
        self.assertTrue(numpy.linalg.norm(K2 - K3.T) < tol)


if __name__ == '__main__':
    unittest.main()

