from apgl.graph import *
from apgl.util import *
import numpy
import logging
from apgl.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor


class CommonNeighboursPredictor(AbstractEdgePredictor):
    """
    Make predictions for a new edge using the Common Neighbours method.
    """
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.printStep = 50

    def learnModel(self, graph):
        Parameter.checkInt(self.windowSize, 1, graph.getNumVertices())
        self.graph = graph


    def predictEdges(self, vertexIndices):
        """
        This makes a prediction for a series of edges using the following score
        |n(x) \cup n(y)|.
        Returns a matrix with rows are a ranked list of verticies of length windowSize.
        """

        logging.info("Running predictEdges in " + str(self.__class__.__name__))

        P = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        S = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        W = self.graph.getWeightMatrix()


        for i in range(vertexIndices.shape[0]):
            Util.printIteration(i, 50, vertexIndices.shape[0])
            scores = numpy.zeros(self.graph.getNumVertices())

            #Maybe something like this:
            #WI = W[vertexIndices[i], :] + W
            #WU = W[vertexIndices[i], :] * W

            for j in range(0, self.graph.getNumVertices()):
                scores[j] = numpy.nonzero(W[vertexIndices[i], :] * W[j, :])[0].shape[0]

            P[i, :], S[i, :] = self.indicesFromScores(vertexIndices[i], scores)

        return P, S

