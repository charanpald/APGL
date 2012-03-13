'''
Created on 3 Aug 2009

@author: charanpal
'''
import unittest
import numpy
from apgl.util.Util import Util
from apgl.data.SparseCenterer import SparseCenterer
from apgl.data.ExamplesList import ExamplesList
from apgl.kernel.LinearKernel import LinearKernel


class PreprocessorTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCenter(self):
        numExamples = 10
        numFeatures = 30
        X = numpy.random.rand(numExamples, numFeatures)
        K = numpy.dot(X, X.T)

        kernel = LinearKernel()
        c = 10 

        sparseCenterer = SparseCenterer()
        KTilde = sparseCenterer.centerArray(X, kernel, c)

        j = numpy.ones((numExamples, 1))

        KTilde2 = K - Util.mdot(j, j.T, K)/numExamples - Util.mdot(K, j, j.T)/numExamples + Util.mdot(j.T, K, j)*numpy.ones((numExamples, numExamples))/(numExamples**2)

        self.assertTrue(numpy.linalg.norm(KTilde-KTilde2) < 0.001)

        #Now test low rank case
        c = 8 
        KTilde = sparseCenterer.centerArray(X, kernel, c)

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

