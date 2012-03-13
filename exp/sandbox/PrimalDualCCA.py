

import numpy
import scipy.linalg
from apgl.util.Parameter import Parameter
from apgl.kernel import *
from apgl.util.Util import Util 

"""
An implementation of the primal-dual Canonincal Correlation Analysis algorithm.
"""

class PrimalDualCCA(object):
    def __init__(self, kernelX, tau1, tau2):
        Parameter.checkFloat(tau1, 0.0, 1.0)
        Parameter.checkFloat(tau2, 0.0, 1.0)
        Parameter.checkClass(kernelX, AbstractKernel)

        self.kernelX = kernelX
        self.tau1 = tau1
        self.tau2 = tau2

    def learnModel(self, X, Y):
        """
        Learn the CCA primal-dual directions.
        """
        self.trainX = X
        self.trainY = Y

        numExamples = X.shape[0]
        numFeatures = Y.shape[1]

        a = 10**-5
        I = numpy.eye(numExamples)
        I2 = numpy.eye(numFeatures)
        Kx = self.kernelX.evaluate(X, X) + a*I
        Kxx = numpy.dot(Kx, Kx)
        Kxy = numpy.dot(Kx, Y) 
        Cyy = numpy.dot(Y.T, Y) + a*I2

        Z1 = numpy.zeros((numExamples, numExamples))
        Z2 = numpy.zeros((numFeatures, numFeatures))
        Z3 = numpy.zeros((numExamples, numFeatures))

        #Note we add a small value to the diagonal of A and B to deal with low-rank
        A = numpy.c_[Z1, Kxy]
        A1 = numpy.c_[Kxy.T, Z2]
        A = numpy.r_[A, A1]
        A = (A+A.T)/2 #Stupid stupidness 

        B = numpy.c_[(1-self.tau1)*Kxx - self.tau1*Kx, Z3]
        B1 = numpy.c_[Z3.T, (1-self.tau2)*Cyy - self.tau2*I2]
        B = numpy.r_[B, B1]
        B = (B+B.T)/2

        (D, W) = scipy.linalg.eig(A, B)

        #Only select eigenvalues which are greater than zero
        W = W[:, D>0]

        #We need to return those eigenvectors corresponding to positive eigenvalues
        self.alpha = W[0:numExamples, :]
        self.V = W[numExamples:, :]
        self.lmbdas = D[D>0]

        alphaDiag = Util.mdot(self.alpha.T, Kxx, self.alpha)
        alphaDiag = alphaDiag + numpy.array(alphaDiag < 0, numpy.int)
        vDiag = Util.mdot(self.V.T, Cyy, self.V)
        vDiag = vDiag + numpy.array(vDiag < 0, numpy.int)
        self.alpha = numpy.dot(self.alpha, numpy.diag(1/numpy.sqrt(numpy.diag(alphaDiag))))
        self.V = numpy.dot(self.V, numpy.diag(1/numpy.sqrt(numpy.diag(vDiag))))

        return self.alpha, self.V, self.lmbdas

    def project(self, testX, testY, k=None):
        if k==None:
            k = self.alpha.shape[1]

        testTrainKx = self.kernelX.evaluate(testX, self.trainX)

        return numpy.dot(testTrainKx, self.alpha[:, 0:k]), numpy.dot(testY, self.V[:, 0:k])

