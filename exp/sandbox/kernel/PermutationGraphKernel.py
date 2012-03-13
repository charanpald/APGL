'''
Created on 26 Jun 2009

@author: charanpal
'''
import scipy.sparse.lil
import numpy
from apgl.kernel.GraphKernel import GraphKernel
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util

class PermutationGraphKernel(GraphKernel):
    def __init__(self, tau, vertexKernel):
        Parameter.checkFloat(tau, 0.0, 1.0)

        self.tau = tau
        self.vertexKernel = vertexKernel

    #For now, assume that graphs are the same size 
    def evaluate(self, g1, g2, debug=False):
        """
        Find the kernel evaluation between two graphs
        """
        #W1 is always the smallest graph
        if g1.getNumVertices() > g2.getNumVertices():
            return self.evaluate(g2, g1)

        #We ought to have something that makes the matrices the same size 
        W1, W2 = self.__getWeightMatrices(g1, g2)
        K1, K2 = self.__getKernelMatrices(g1, g2)

        #Find common eigenspace
        S1, U = numpy.linalg.eigh(self.tau*W1 + (1-self.tau)*K1)
        S2, V = numpy.linalg.eigh(self.tau*W2 + (1-self.tau)*K2)

        #Find appoximate diagonals
        SK1 = numpy.diag(Util.mdot(U.T, K1, U))
        SW1 = numpy.diag(Util.mdot(U.T, W1, U))
        SK2 = numpy.diag(Util.mdot(V.T, K2, V))
        SW2 = numpy.diag(Util.mdot(V.T, W2, V))

        evaluation = self.tau * numpy.dot(SW1, SW2) + (1-self.tau)*numpy.dot(SK1, SK2)
        
        if debug:
            P = numpy.dot(V, U.T)
            f = self.getObjectiveValue(self.tau, P, g1, g2)
            return (evaluation, f, P, SW1, SW2, SK1, SK2)
        else:
            return evaluation

    def getObjectiveValue(self, tau, P, g1, g2):
        W1, W2 = self.__getWeightMatrices(g1, g2)
        K1, K2 = self.__getKernelMatrices(g1, g2)

        f = tau * numpy.linalg.norm(Util.mdot(P, W1, P.T) - W2) + (1-tau)* numpy.linalg.norm(Util.mdot(P, K1, P.T) - K2)
        return f 
        
    def __getKernelMatrices(self, g1, g2):
        X1, X2 = self.__getVertexMatrices(g1, g2)
        K1 = self.vertexKernel.evaluate(X1, X1)
        K2 = self.vertexKernel.evaluate(X2, X2)
        return K1, K2

    def __getWeightMatrices(self, g1, g2):
        if type(g1.getWeightMatrix()) == scipy.sparse.lil_matrix:
            W1 = g1.getWeightMatrix().todense()
        else:
            W1 = g1.getWeightMatrix()

        if type(g2.getWeightMatrix()) == scipy.sparse.lil_matrix:
            W2 = g2.getWeightMatrix().todense()
        else:
            W2 = g2.getWeightMatrix()

        n1 = W1.shape[0]
        n2 = W2.shape[0]

        W1Ext = numpy.zeros((n2, n2))
        W1Ext[numpy.ix_(list(range(0,n1)), list(range(0,n1)))] = W1
        return W1Ext, W2 

    def __getVertexMatrices(self, g1, g2):
        X1 = g1.getVertexList().getVertices(list(range(0, g1.getNumVertices())))
        X2 = g2.getVertexList().getVertices(list(range(0, g2.getNumVertices())))

        n1 = X1.shape[0]
        n2 = X2.shape[0]
        numFeatures = X1.shape[1]

        #Extend X1 with zeros so that it is the same size as X2
        X1 = numpy.append(X1, numpy.zeros((n2 - n1, numFeatures)), 0)
        return X1, X2
