
import numpy
import scipy.linalg
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util 

"""
An implementation of the Primal PCA algorithm. 
"""

class PrimalPCA(object):
    def __init__(self, k=10):
        """
        Create the PCA object with number of projection directions k
        """
        self.setK(k)

    def setK(self, k):
        Parameter.checkInt(k, 1, float('inf'))
        self.k = k

    def getK(self):
        return self.k

    def learnModel(self, X):
        """
        Learn the PCA directions.

        :param X: The training examples organised into rows.
        :type X: :class:`numpy.ndarray`
        """
        Parameter.checkClass(X, numpy.ndarray)

        C = numpy.dot(X.T, X)
        (lmbdas, U) = scipy.linalg.eig(C)

        inds = numpy.flipud(numpy.argsort(lmbdas)) 

        self.lmbdas = lmbdas[inds]
        self.U = U[:, inds]

        return self.U, self.lmbdas

    def project(self, testX):
        """
        Project the examples into the PCA space using k directions

        :param testX: The examples to project given as rows of a matrix.
        :type testX: :class:`numpy.ndarray`

        :returns: The projected examples.
        """
        Parameter.checkClass(testX, numpy.ndarray)
        return numpy.dot(testX, self.U[:, 0:self.k])

