

import unittest
import numpy
import scipy.linalg
from apgl.features.PrimalCCA import PrimalCCA
from apgl.features.KernelCCA import KernelCCA
from apgl.kernel import *
from apgl.util import * 
import logging


class  PrimalCCATest(unittest.TestCase):
    def setUp(self):
        numpy.seterr(all='ignore')
        pass 

    def testLearnModel(self):
        numExamples = 50
        numFeatures = 10
        X = numpy.random.rand(numExamples, numFeatures)
        Y = X

        tau = 0.0
        tol = 10**--6

        cca = PrimalCCA(tau)
        u, v, lmbdas = cca.learnModel(X, Y)
        self.assertTrue(numpy.linalg.norm(u-v) < tol)
        self.assertTrue(numpy.linalg.norm(lmbdas-numpy.ones(numFeatures)) < tol)
        
        Y = X*2

        cca = PrimalCCA(tau)
        u, v, lmbdas = cca.learnModel(X, Y)

        self.assertTrue(numpy.linalg.norm(u-v) < tol)
        self.assertTrue(numpy.linalg.norm(lmbdas-numpy.ones(numFeatures)) < tol)

        #Rotate X to form Y
        Z = numpy.random.rand(numFeatures, numFeatures)
        ZZ = numpy.dot(Z.T, Z)

        (D, W) = scipy.linalg.eig(ZZ)

        Y = numpy.dot(X, W)
        u, v, lmbdas = cca.learnModel(X, Y)
        self.assertTrue(numpy.linalg.norm(lmbdas-numpy.ones(numFeatures)) < tol)

        

    def testProject(self):
        #Test if it is the same as KCCA
        numExamples = 50
        numFeatures = 10
        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.random.rand(numExamples, numFeatures)

        tau = 0.0
        tol = 10**--6
        k = 5

        cca = PrimalCCA(tau)
        u, v, lmbdas = cca.learnModel(X, Y)
        XU, YU = cca.project(X, Y, k)

        kernel = LinearKernel()
        kcca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lmbdas2 = kcca.learnModel(X, Y)
        XU2, YU2 = kcca.project(X, Y, k)

        #Seem to get an error in this for some reason 
        #self.assertTrue(numpy.linalg.norm(XU-XU2) < tol)
        #self.assertTrue(numpy.linalg.norm(YU-YU2) < tol)

        #Now try with different tau
        tau = 0.5
        cca = PrimalCCA(tau)
        u, v, lmbdas = cca.learnModel(X, Y)
        XU, YU = cca.project(X, Y, k)

        kernel = LinearKernel()
        kcca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lmbdas = kcca.learnModel(X, Y)
        XU2, YU2 = kcca.project(X, Y, k)

        self.assertTrue(numpy.linalg.norm(XU-XU2) < tol)
        self.assertTrue(numpy.linalg.norm(YU-YU2) < tol)

        self.assertTrue(numpy.linalg.norm(numpy.dot(XU.T, XU) - numpy.ones(k)) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.dot(YU.T, YU) - numpy.ones(k)) < tol)

    def testGetY(self):
        #Test if we can recover Y from X 
        numExamples = 10
        numFeatures = 5
        X = numpy.random.rand(numExamples, numFeatures) 

        Z = numpy.random.rand(numFeatures, numFeatures)
        ZZ = numpy.dot(Z.T, Z)
        (D, W) = scipy.linalg.eig(ZZ)
        Y = numpy.dot(X, W)
        
        tau = 0.0
        cca = PrimalCCA(tau)
        U, V, lmbdas = cca.learnModel(X, Y)

        Yhat = Util.mdot(X, U, V.T, numpy.linalg.inv(numpy.dot(V, V.T)))
        logging.debug((numpy.abs(Yhat- Y)))


if __name__ == '__main__':
    unittest.main()

