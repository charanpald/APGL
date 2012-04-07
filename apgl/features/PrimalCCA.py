

import numpy
import scipy.linalg
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util 

"""
An implementation of the primal Canonincal Correlation Analysis (CCA) algorithm.
"""

class PrimalCCA(object):
    def __init__(self, tau):
        """
        Intialise the object with regularisation parameter tau between 0 (no
        regularisation) and 1 (full regularisation). 
        """
        Parameter.checkFloat(tau, 0.0, 1.0)

        self.tau = tau

    def learnModel(self, X, Y):
        """
        Learn the CCA  directions using set of examples given the numpy.ndarrays
        X and Y. These matrices have rows as their example, and must have the same
        number of rows. 
        """
        numFeatures = X.shape[1]

        I = numpy.eye(numFeatures)
        Z = numpy.zeros((numFeatures, numFeatures))
        Cxx = numpy.dot(X.T, X)
        Cxy = numpy.dot(X.T, Y)
        Cyy = numpy.dot(Y.T ,Y)

        A = numpy.c_[Z, Cxy]
        A1 = numpy.c_[Cxy.T, Z]
        A = numpy.r_[A, A1]

        B = numpy.c_[(1-self.tau)*Cxx - self.tau*I, Z]
        B1 = numpy.c_[Z, (1-self.tau)*Cyy - self.tau*I]
        B = numpy.r_[B, B1]

        (D, W) = scipy.linalg.eig(A, B)

        #Only select eigenvalues which are greater than zero
        W = W[:, D>0]

        #We need to return those eigenvectors corresponding to positive eigenvalues
        self.U = W[0:numFeatures, :]
        self.V = W[numFeatures:numFeatures*2, :]
        self.lmbdas = D[D>0]

        self.U = numpy.dot(self.U, numpy.diag(1/numpy.sqrt(numpy.diag(self.U.T.dot(Cxx).dot(self.U)))))
        self.V = numpy.dot(self.V, numpy.diag(1/numpy.sqrt(numpy.diag(self.V.T.dot(Cyy).dot(self.V)))))

        return self.U, self.V, self.lmbdas

    def project(self, testX, testY, k=None):
        """
        Project the examples in the CCA subspace using set of test examples testX
        and testY. The number of projection directions is specified with k, and
        if this parameter is None then all directions are used. 
        """
        if k==None:
            k = self.U.shape[1]

        return numpy.dot(testX, self.U[:, 0:k]), numpy.dot(testY, self.V[:, 0:k])