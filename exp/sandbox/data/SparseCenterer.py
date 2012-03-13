import numpy
from apgl.util.Parameter import Parameter 
from apgl.util.Util import Util 

class SparseCenterer():
    def __init__(self):
        pass

    def centerArray(self, X, kernel, c, returnAlpha=False):
        """
        A method to center a kernel matrix.
        """
        #Note the following check does not ensure the inverse exists
        Parameter.checkInt(c, 0, X.shape[0])
        numExamples = X.shape[0]
        K = kernel.evaluate(X, X)
        j = numpy.ones((numExamples, 1))

        #inds = self.__chooseRandomIndices(X, kernel, c)
        inds = self.__chooseIndices(X, kernel, c)
        epsilon = 0.01

        alphaT = Util.mdot(numpy.linalg.inv(K[numpy.ix_(inds, inds)] + epsilon*numpy.eye(c)), K[inds, :], j)/numExamples

        KTilde = K - Util.mdot(j, alphaT.T, K[inds, :]) -  Util.mdot(K[:, inds], alphaT, j.T) + Util.mdot(alphaT.T, K[numpy.ix_(inds, inds)], alphaT)*numpy.ones((numExamples, numExamples))

        if returnAlpha:
            alpha = numpy.zeros((numExamples, 1))
            alpha[inds, :] = alphaT
            return KTilde, alpha
        else:
            return KTilde 

    def __chooseRandomIndices(self, X, kernel, c):
        numExamples = X.shape[0]
        inds = numpy.random.permutation(numExamples)
        inds = inds[0:c]
        return inds 

    def __chooseIndices(self, X, kernel, c):
        numExamples = X.shape[0]
        K = kernel.evaluate(X, X)
        j = numpy.ones((numExamples, 1))

        phis = numpy.zeros(numExamples)

        for j in range(numExamples):
            phis[j] = -numpy.sum(K[:, j])**2/(K[j,j])

        inds = numpy.argsort(phis)
        #print(phis[inds])
        return inds[0:c]