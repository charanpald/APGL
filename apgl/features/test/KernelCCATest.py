
import unittest
import numpy
import scipy.linalg
from apgl.features.KernelCCA import KernelCCA
from apgl.kernel.LinearKernel import LinearKernel

class  KernelCCATest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(precision=3, suppress=True)
        numpy.random.seed(21)

    def testLearnModel(self):
        numExamples = 5
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        Y = X

        tau = 0.0
        kernel = LinearKernel()

        tol = 10**--6

        cca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lmbdas = cca.learnModel(X, Y)

        self.assertTrue(numpy.linalg.norm(alpha-beta) < tol)
        self.assertTrue(numpy.linalg.norm(lmbdas-numpy.ones(numExamples)) < tol)

        Y = X*2

        cca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lmbdas = cca.learnModel(X, Y)

        self.assertTrue(numpy.linalg.norm(alpha-beta) < tol)
        self.assertTrue(numpy.linalg.norm(lmbdas-numpy.ones(numExamples)) < tol)

        #Test case with tau = 1 is the same as 1st KPLS direction 
        Y = numpy.random.rand(numExamples, numFeatures)
        tau = 1.0

        cca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lmbdas = cca.learnModel(X, Y)

        Kx = kernel.evaluate(X, X)
        Ky = kernel.evaluate(Y, Y)

        lmbdas2, alpha2 = scipy.linalg.eig(numpy.dot(Ky, Kx))

        alpha = alpha[:, numpy.argmax(lmbdas)]/numpy.linalg.norm(alpha[:, numpy.argmax(lmbdas)])
        alpha2 = alpha2[:, numpy.argmax(lmbdas2)]/numpy.linalg.norm(alpha2[:, numpy.argmax(lmbdas2)])

        self.assertTrue(numpy.linalg.norm(alpha-alpha2) < tol)


    def testProject(self):
        numExamples = 5
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        Y = X*2

        tau = 0.0
        kernel = LinearKernel()

        tol = 10**--6
        k = 5

        cca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lmbdas = cca.learnModel(X, Y)

        XU, YU = cca.project(X, Y, k)

        self.assertTrue(numpy.linalg.norm(XU-YU) < tol)

        self.assertTrue(numpy.linalg.norm(numpy.dot(XU.T, XU) - numpy.ones(k)) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.dot(YU.T, YU) - numpy.ones(k)) < tol)
if __name__ == '__main__':
    unittest.main()

