import numpy
from apgl.kernel.AbstractKernel import AbstractKernel
from apgl.util.Parameter import Parameter 

class LinearKernel(AbstractKernel):
    """
    A class to find linear kernel evaluations k(x, y) = <x, y> 
    """
    def __init__(self):
        """
        Intialise class. 
        """
        pass

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

        return numpy.dot(X1, X2.T)