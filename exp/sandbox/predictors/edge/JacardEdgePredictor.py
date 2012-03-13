from apgl.graph import *
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor
import numpy
import logging


class JacardEdgePredictor(AbstractEdgePredictor):
    """
    Make predictions for a new edge using the Jacard measure. 
    """
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.printStep = 50

    def learnModel(self, graph):
        Parameter.checkInt(self.windowSize, 1, graph.getNumVertices())
        self.graph = graph

    def predictEdges(self, vertexIndices):
        """
        This makes a prediction for a series of edges using the Jacard Index.
        Returns a matrix with rows are a ranked list of verticies of length windowSize.
        """

        """
        The score is the |n(x) \cup n(y)|/|n(x) \cap n(y)|. This is faster than
        the other method. 
        """
        logging.info("Running predictEdges in " + str(self.__class__.__name__))
        printStep = 50 

        P = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        S = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        W = self.graph.getWeightMatrix()


        for i in range(vertexIndices.shape[0]):
            Util.printIteration(i, printStep, vertexIndices.shape[0])
            scores = numpy.zeros(self.graph.getNumVertices())

            #Maybe something like this: 
            #WI = W[vertexIndices[i], :] + W
            #WU = W[vertexIndices[i], :] * W

            for j in range(0, self.graph.getNumVertices()):
                scores[j] = numpy.nonzero(W[vertexIndices[i], :] + W[j, :])[0].shape[0]

                if scores[j] != 0:
                    scores[j] = numpy.nonzero(W[vertexIndices[i], :] * W[j, :])[0].shape[0]/float(scores[j])

            
            P[i, :], S[i, :] = self.indicesFromScores(vertexIndices[i], scores)

        return P, S

