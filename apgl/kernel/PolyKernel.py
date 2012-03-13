
import numpy

from apgl.kernel.AbstractKernel import AbstractKernel
from apgl.util.Parameter import Parameter

class PolyKernel(AbstractKernel):
    """
    A class to find polynomial kernel evaluations k(x, y) = (<x, y> + b)^degree
    """
    def __init__(self, b=1.0, degree=2):
        """
        Initialise object with given value of b >= 0 and degree

        :param b: kernel bias parameter.
        :type b: :class:`float`

        :param degree: degree parameter.
        :type degree: :class:`int`
        """
        self.setB(b)
        self.setDegree(degree)

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

        K = (numpy.dot(X1, X2.T) + self.b)**self.degree

        return K

    def setB(self, b):
        """
        Set the b parameter.

        :param b: kernel bias parameter.
        :type b: :class:`float`
        """
        Parameter.checkFloat(b, 0.0, float('inf'))

        self.b = b

    def setDegree(self, degree):
        """
        Set the degree parameter.

        :param degree: kernel degree parameter.
        :type degree: :class:`int`
        """
        Parameter.checkInt(degree, 1, float('inf'))

        self.degree = degree