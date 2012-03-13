
from apgl.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
import numpy
import logging

class EgoEdgePredictor(AbstractEdgePredictor):
    """
    A class which splits the graph into ego networks and then makes predictions
    assuming that all ego networks are independent.
    """

    def __init__(self, alterRegressor, egoRegressor, windowSize):
        """
        The alterRegressor must be a primal method, since the number of alters
        for each ego vary, and hence the dual vectors are not constant in size. 
        """
        Parameter.checkClass(alterRegressor, AbstractPredictor)
        Parameter.checkClass(egoRegressor, AbstractPredictor)
        
        self.alterRegressor = alterRegressor
        self.egoRegressor = egoRegressor
        self.windowSize = windowSize


    def learnModel(self, graph):
        """
        Learn a prediction model based on all of the edges of the input graph.
        For each ego, X contains a list of neighbours and non-neighbours in the same
        ratio, and y = 1 when for a neighbour otherwise -1. We then find the set of
        primal weights w for each ego network and then regress onto the set of weights
        using the ego labels.

        One can either learn by comparing neighbours and non-neighbours, or alternatively
        using the labels of edges and making prediction on unlabelled edges. 

        :param graph: The input graph to learn from.
        :type graph: class:`apgl.graph.AbstractSingleGraph`

        :param randomNegLabel: How to compute edge labels, False means use the labels
        themselves, and True means randomly pick non-neighbours to have -1 labels
        :type randomNegLabel: class `bool`
        """

        Parameter.checkInt(self.windowSize, 1, graph.getNumVertices())
        self.graph = graph
        logging.info("Learning model on graph of size " + str(graph.getNumVertices()))

        allIndices = numpy.arange(0, graph.getNumVertices())
        V = graph.getVertexList().getVertices(allIndices)
        W = numpy.zeros((0, graph.getVertexList().getNumFeatures()))
        Xe  =  numpy.zeros((0, graph.getVertexList().getNumFeatures()))
        printStep = numpy.floor(graph.getNumVertices()/10)

        for i in range(graph.getNumVertices()):
            Util.printIteration(i, printStep, graph.getNumVertices())
            neighbours = graph.neighbours(i)

            if neighbours.shape[0] != 0:
                compNeighbours = numpy.setdiff1d(allIndices, neighbours)
                perm = numpy.random.permutation(compNeighbours.shape[0])[0:neighbours.shape[0]]
                negativeVertices = V[compNeighbours[perm], :]
                X = numpy.r_[V[neighbours, :], negativeVertices]
                y = numpy.ones(X.shape[0])
                y[neighbours.shape[0]:] = -1
 
                w = self.alterRegressor.learnModel(X, y)
                W = numpy.r_[W, numpy.array([w])]
                Xe = numpy.r_[Xe, numpy.array([V[i, :]])]

        #Now we need to solve least to find regressor of Xe onto W
        self.egoRegressor.learnModel(Xe, W)


    def predictEdges(self, vertexIndices):
        """
        Make prediction for all posible edges given the vertex indices 
        """
        Parameter.checkInt(self.graph.getVertexList().getNumFeatures(), 1, float('inf'))
        logging.info("Making prediction over " + str(vertexIndices.shape[0]) + " vertices")

        testX = self.graph.getVertexList().getVertices(vertexIndices)
        testW = self.egoRegressor.predict(testX)

        #Output scores of resulting vertices
        V = self.graph.getVertexList().getVertices(list(range(0, self.graph.getNumVertices())))
        P = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        S = numpy.zeros((vertexIndices.shape[0], self.windowSize))

        for i in range(testX.shape[0]):
            scores = numpy.dot(V, testW[i, :])
            P[i, :], S[i, :] = self.indicesFromScores(vertexIndices[i], scores) 

        return P, S
