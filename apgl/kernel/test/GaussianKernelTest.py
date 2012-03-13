
import unittest
import numpy 
from apgl.kernel.GaussianKernel import GaussianKernel

class  GaussianKernelTest(unittest.TestCase):
    def setUp(self):
        pass

    def testEvaluate(self):
        numExamples = 10
        numFeatures = 5

        X = numpy.random.randn(numExamples, numFeatures)

        tol = 10**-6 
        sigma = 1.0 
        kernel = GaussianKernel(sigma)

        K = kernel.evaluate(X, X)

        K2 = numpy.zeros((numExamples, numExamples))

        for i in range(numExamples):
            for j in range(numExamples):
                K2[i, j] = numpy.exp(-numpy.linalg.norm(X[i, :] - X[j, :])**2/2*sigma**2)

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
                K2[i, j] = numpy.exp(-numpy.linalg.norm(X[i, :] - X2[j, :])**2/2*sigma**2)


        self.assertTrue(numpy.linalg.norm(K2 - K) < tol)
        self.assertTrue(numpy.linalg.norm(K2 - K3.T) < tol)


if __name__ == '__main__':
    unittest.main()

