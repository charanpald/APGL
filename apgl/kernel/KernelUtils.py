
import numpy 

class KernelUtils():
    def __init__(self):
        pass

    @staticmethod 
    def computeDistanceMatrix(K):
        """
        Compute a matrix of distances from a kernel matrix. Each entry is
        D_ij = \|x_i - x_j\|
        """
        if K.shape[0] != K.shape[1]:
            raise ValueError("Kernel matrix must be square")

        numExamples = K.shape[0]
        j = numpy.ones((numExamples, 1))
        dK = numpy.array([numpy.diag(K)]).T

        D = numpy.dot(dK, j.T) - 2*K + numpy.dot(j, dK.T)
        D = D ** 0.5

        return D 

