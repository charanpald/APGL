from apgl.graph import *
from apgl.util import *
from apgl.sandbox.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor
import numpy
import logging

class RandomEdgePredictor(AbstractEdgePredictor):
    """
    Make predictions for a new edge using the random numbers. 
    """
    def __init__(self, windowSize):
        self.windowSize = windowSize

    def learnModel(self, graph):
        self.graph = graph
        Parameter.checkInt(self.windowSize, 1, graph.getNumVertices())

    """
    The following two functions are just dummy ones to test the cvModelSelection
    method. 
    """
    def setC(self, C):
        #print(("C="+str(C)))
        pass

    def setD(self, D):
        #print(("D="+str(D)))
        pass 

    def predictEdges(self, vertexIndices):
        """
        This makes a prediction for a series of edges using random permutations.
        Returns a matrix with rows are a ranked list of verticies of length windowSize.
        """

        Parameter.checkInt(self.windowSize, 1, self.graph.getNumVertices())
        logging.info("Running predictEdges in " + str(self.__class__.__name__))

        P = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        S = numpy.zeros((vertexIndices.shape[0], self.windowSize))

        for i in range(vertexIndices.shape[0]):
            scores = numpy.random.randn(self.graph.getNumVertices())
            P[i, :], S[i, :] = self.indicesFromScores(vertexIndices[i], scores)

        return P, S

