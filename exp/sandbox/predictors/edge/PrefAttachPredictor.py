from apgl.graph import *
from apgl.util import *
from apgl.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor
import numpy
import logging


class PrefAttachPredictor(AbstractEdgePredictor):
    """
    Make predictions for a new edge using preferencial attachment.
    """
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.printStep = 50

    def learnModel(self, graph):
        Parameter.checkInt(self.windowSize, 1, graph.getNumVertices())
        self.graph = graph


    def predictEdges(self, vertexIndices):
        """
        This makes a prediction for a series of edges using preferencial attachment.
        Returns a matrix with rows are a ranked list of verticies of length windowSize. 
        """

        """
        The score is the degree of x times the degree of y.
        """
        logging.info("Running predictEdges in " + str(self.__class__.__name__))

        P = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        S = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        degrees = numpy.zeros(self.graph.getNumVertices())
        W = self.graph.getSparseWeightMatrix()

        for i in range(0, W.shape[0]):
            degrees[i] = W[i, :].getnnz()

        for i in range(vertexIndices.shape[0]):
            scores = degrees * degrees[vertexIndices[i]]
            P[i, :], S[i, :] = self.indicesFromScores(vertexIndices[i], scores)

        return P, S
        


            