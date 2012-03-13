from apgl.graph import *
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor
import numpy
import logging

class AdamicPredictor(AbstractEdgePredictor):
    """
    Make predictions for a new edge using the Adamic/Adar method.
    """
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.printStep = 50 

    def learnModel(self, graph):
        self.graph = graph


    def predictEdges(self, vertexIndices):
        """
        This makes a prediction for a series of edges using the following score
        \sum_z \in n(x) \cup n(y) = 1/|log(n(z)|
        Returns a matrix with rows are a ranked list of verticies of length self.windowSize.
        """

        Parameter.checkInt(self.windowSize, 1, self.graph.getNumVertices())
        logging.info("Running predictEdges in " + str(self.__class__.__name__))

        P = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        S = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        W = self.graph.getWeightMatrix()


        for i in range(vertexIndices.shape[0]):
            Util.printIteration(i, self.printStep, vertexIndices.shape[0])
            scores = numpy.zeros(self.graph.getNumVertices())

            for j in range(0, self.graph.getNumVertices()):
                commonNeighbours = numpy.nonzero(W[vertexIndices[i], :] * W[j, :])[0]

                for k in commonNeighbours:
                    q = numpy.log(numpy.nonzero(W[k, :])[0].shape[0])
                    if q != 0:
                        scores[j] = scores[j] + 1/q


            P[i, :], S[i, :] = self.indicesFromScores(vertexIndices[i], scores)

        return P, S

