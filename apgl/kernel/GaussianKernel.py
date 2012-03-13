
import numpy
 
from apgl.kernel.AbstractKernel import AbstractKernel
from apgl.util.Parameter import Parameter 

class GaussianKernel(AbstractKernel):
    """
    A class to find gaussian kernel evaluations k(x, y) = exp (-||x - y||^2/2 sigma^2)
    """
    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.setSigma(sigma)

    def evaluate(self, X1, X2):
        """
        Find kernel evaluation between two matrices X1 and X2 whose rows are
        examples and have an identical number of columns.


        :param X1: First set of examples.
        :type X1: :class:`numpy.ndarray`

        :param X2: Second set of examples.
        :type X2: :class:`numpy.ndarray`
        """
        Parameter.checkClass(X1, numpy.ndarray)
        Parameter.checkClass(X2, numpy.ndarray)
        
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        j1 = numpy.ones((X1.shape[0], 1))
        j2 = numpy.ones((X2.shape[0], 1))

        diagK1 = numpy.sum(X1**2, 1)
        diagK2 = numpy.sum(X2**2, 1)

        X1X2 = numpy.dot(X1, X2.T)

        Q = (2*X1X2 - numpy.outer(diagK1, j2) - numpy.outer(j1, diagK2) )/ (2*self.sigma**2)

        return numpy.exp(Q)

    def setSigma(self, sigma):
        """
        Set the sigma parameter.

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        Parameter.checkFloat(sigma, 0.0, float('inf'))

        if sigma == 0.0:
            raise ValueError("Sigma cannot be zero")

        self.sigma = sigma

    def __str__(self):
        return "GaussianKernel: sigma = " + str(self.sigma)