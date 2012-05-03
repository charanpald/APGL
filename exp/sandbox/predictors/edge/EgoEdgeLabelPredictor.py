

import logging
import gc 

from apgl.predictors.AbstractPredictor import AbstractPredictor
from exp.sandbox.predictors.edge.AbstractEdgeLabelPredictor import AbstractEdgeLabelPredictor
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.data import * 
import numpy

class EgoEdgeLabelPredictor(AbstractEdgeLabelPredictor):
    """
    A class which splits the graph into ego networks and then makes predictions
    assuming that all ego networks are independent.
    """

    def __init__(self, alterRegressor, egoRegressor):
        """
        The alterRegressor must be a primal method, since the number of alters
        for each ego vary, and hence the dual vectors are not constant in size.
        """
        Parameter.checkClass(alterRegressor, AbstractPredictor)
        Parameter.checkClass(egoRegressor, AbstractPredictor)

        self.alterRegressor = alterRegressor
        self.egoRegressor = egoRegressor

    def learnModel(self, graph):
        """
        Learn a prediction model based on considering ego networks as independent.
        For each ego, X contains a list of neighbours and the corresponding labels
        are the values of the edge labels. We then find the set of primal weights
        w for each ego network and then regress onto the set of weights using the
        ego labels.

        :param graph: The input graph to learn from.
        :type graph: class:`apgl.graph.AbstractSingleGraph`
        """

        logging.info("Learning model on graph of size " + str(graph.getNumVertices()))
        logging.info("EgoLearner: " + str(self.egoRegressor))
        logging.info("AlterLearner: " + str(self.alterRegressor))

        allIndices = numpy.arange(0, graph.getNumVertices())
        V = graph.getVertexList().getVertices(list(allIndices))
        W = numpy.zeros((0, graph.getVertexList().getNumFeatures()))
        Xe  =  numpy.zeros((0, graph.getVertexList().getNumFeatures()))
        printStep = numpy.floor(graph.getNumVertices()/10)
        alterError = 0.0 

        for i in range(graph.getNumVertices()):
            Util.printIteration(i, printStep, graph.getNumVertices())
            neighbours = graph.neighbours(i)

            if neighbours.shape[0] != 0:
                X = V[neighbours, :]
                y = numpy.ones(X.shape[0])

                for j in range(neighbours.shape[0]):
                    y[j] = graph.getEdge(i, neighbours[j])


                w = self.alterRegressor.learnModel(X, y)
                #alterError = numpy.mean(numpy.abs(self.alterRegressor.predict(X) - y))

                W = numpy.r_[W, numpy.array([w])]
                Xe = numpy.r_[Xe, numpy.array([V[i, :]])]

        #Now we need to solve least to find regressor of Xe onto W
        logging.info("Finding regression matrix onto weights using matrix of size " + str(Xe.shape))
        gc.collect()
        #self.standardiser = Standardiser()
        #self.standardiser2 = Standardiser()
        #Xe = self.standardiser.standardiseArray(Xe)
        #W = self.standardiser2.standardiseArray(W)
        self.egoRegressor.learnModel(Xe, W)


        return W 


    def predictEdges(self, graph, edges):
        """
        Make prediction  given the edges and given graph.

        :param edges: A numpy array consisting of the edges to make predictions over.
        """
        Parameter.checkInt(graph.getVertexList().getNumFeatures(), 1, float('inf'))
        logging.info("Making prediction over " + str(edges.shape[0]) + " edges")

        predY = numpy.zeros(edges.shape[0])

        for i in range(edges.shape[0]):
            #Make a prediction for each ego-alter 
            egoInd = edges[i, 0]
            alterInd = edges[i, 1]

            ego = numpy.array([graph.getVertex(egoInd)])
            #ego = self.standardiser.standardiseArray(ego)
            c = self.egoRegressor.predict(ego)
            #c = self.standardiser2.unstandardiseArray(c)
            predY[i] = numpy.dot(graph.getVertex(alterInd), c.ravel())

        return predY 

    #TODO: Write this
    def classifyEdges(self, graph, edges):
        pass
