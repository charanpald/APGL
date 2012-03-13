from apgl.graph import *
from apgl.util import * 
import numpy


class EdgePredictor(object):
    """
    This takes a graph and makes predictions for new edge using the method in
    `A regularisation framework for learning from graph data'. 
    """
    def __init__(self):
        pass

    def computeC(self, graph, alpha):
        """
        Computes the matrix C = (I - alpha S)^-1.
        """
        numVertices = graph.getNumVertices()
        W = graph.getWeightMatrix()
        d = numpy.sum(W, 0)
        d = d + numpy.array(d==0, numpy.int32)
        D = numpy.diag(d**-0.5)
        S = Util.mdot(D, W, D)
        
        I = numpy.eye(numVertices)
        C = I - alpha*S 
        C = numpy.linalg.inv(C)

        return C 

    def predictEdge(self, graph, alpha, selfEdges=False):
        """
        Make a prediction for a new edge of a graph using C. 
        """
        C = self.computeC(graph, alpha)
        W = graph.getWeightMatrix()

        #Return best edge that does not exist
        C[W!=0] = 0

        if not selfEdges:
            C = C - numpy.diag(numpy.diag(C))

        return numpy.unravel_index(numpy.argmax(C), C.shape)
