import scipy.sparse.lil
#A class to compute graph kernels

import numpy 
from apgl.kernel.GraphKernel import GraphKernel

class RandWalkGraphKernel(GraphKernel):
    """
    This is an implementation of the marginalised graph kernel give in "Learning to
    find Graph pre-images", Bakir et al.
    """
    def __init__(self, lmbda):
        #The norm of lambda*norm(Wx) should be less than 1 
        self.lmbda = lmbda

    def getWeightMatrix(self, g):
        """
        This just returns the matrix of weights for 1 graph.
        """
        return numpy.kron(g1.getWeightMatrix(), g2.getWeightMatrix())

    def getInitialProbabilties(self, g1, g2):
        p1 = numpy.ones(g1.getNumVertices())/g1.getNumVertices()
        p2 = numpy.ones(g2.getNumVertices())/g2.getNumVertices()

        return numpy.kron(p1, p2)

    def getFinalProbabilities(self, g1, g2):
        p1 = numpy.ones(g1.getNumVertices())*self.lmbda
        p2 = numpy.ones(g2.getNumVertices())*self.lmbda

        return numpy.kron(p1, p2)

    def evaluate(self, g1, g2):
        """
        The edge transition probabilities are computed by normalising the rows of the
        weight matrices so that probabilities sum to 1. Do we include lambda here?
        Only works for connected graphs! 
        """

        #This bit is a bit ugly - maybe have g1.getDenseWeightMatrix()? 
        if type(g1.getWeightMatrix()) == scipy.sparse.lil.lil_matrix:
            W1 = numpy.array(g1.getWeightMatrix().todense())
        else:
            W1 = numpy.array(g1.getWeightMatrix())

        if type(g2.getWeightMatrix()) == scipy.sparse.lil.lil_matrix:
            W2 = numpy.array(g2.getWeightMatrix().todense())
        else:
            W2 = numpy.array(g2.getWeightMatrix())

        P1 = W1 / numpy.array([numpy.sum(W1, 1) + numpy.array(numpy.sum(W1, 1)==0, numpy.float64)]).T
        P2 = W2 / numpy.array([numpy.sum(W2, 1) + numpy.array(numpy.sum(W2, 1)==0, numpy.float64)]).T

        Wx = numpy.kron(P1, P2)
        px = self.getInitialProbabilties(g1, g2)
        qx = self.getFinalProbabilities(g1, g2)

        I = numpy.eye(Wx.shape[0], Wx.shape[1])
        Z = I - self.lmbda*Wx
        
        return numpy.dot(numpy.dot(qx.T, numpy.linalg.inv(Z)), px)
