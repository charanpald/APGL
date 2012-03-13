
import logging
import gc

from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.predictors.edge.AbstractEdgeLabelPredictor import AbstractEdgeLabelPredictor
from apgl.graph.GraphUtils import GraphUtils
from apgl.util import *
import numpy

class FlatEdgeLabelPredictor(AbstractEdgeLabelPredictor):
    """
    A class which just uses all pairs of ego-alter and then a predictor to find
    the function to map vertex pairs to edge labels. E.g. can just use an SVM. 
    """

    def __init__(self, predictor):
        """
        The predictor must map from vectors to floats.
        """
        Parameter.checkClass(predictor, AbstractPredictor)

        self.predictor = predictor

    def learnModel(self, graph):
        """
        Learn a prediction model based on considering all ego-alter pairs. 

        :param graph: The input graph to learn from.
        :type graph: class:`apgl.graph.AbstractSingleGraph`
        """

        logging.info("Learning model on graph of size " + str(graph.getNumVertices()))
        logging.info("Regressor: " + str(self.predictor))

        edges = graph.getAllEdges()

        if graph.isUndirected():
            edges2 = numpy.c_[edges[:, 1], edges[:, 0]]
            edges = numpy.r_[edges, edges2]

        X = GraphUtils.vertexLabelPairs(graph, edges)
        y = graph.getEdgeValues(edges)

        #Now we need to solve least to find regressor of X onto y
        logging.info("Number of vertex pairs " + str(X.shape))
        gc.collect()
        self.predictor.learnModel(X, y)

    def predictEdges(self, graph, edges):
        """
        Make prediction  given the edges and given graph.

        :param edges: A numpy array consisting of the edges to make predictions over.
        """
        Parameter.checkInt(graph.getVertexList().getNumFeatures(), 1, float('inf'))
        logging.info("Making prediction over " + str(edges.shape[0]) + " edges")

        X = GraphUtils.vertexLabelPairs(graph, edges)
        predY = self.predictor.predict(X)

        return predY 