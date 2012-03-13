
import unittest
import numpy
from apgl.kernel.KernelUtils import KernelUtils


class LinearKernelTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testComputeDistanceMatrix(self):
        numExamples = 10
        numFeatures = 10 
        X = numpy.random.rand(numExamples, numFeatures)
        K = numpy.dot(X, X.T)

        D1 = KernelUtils.computeDistanceMatrix(K)
        D2 = numpy.zeros((numExamples, numExamples))

        for i in range(0, numExamples):
            for j in range(0, numExamples):
                D2[i, j] = numpy.linalg.norm(X[i, :] - X[j, :])

        self.assertTrue(numpy.linalg.norm(D1 - D2) < 10**-6)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
