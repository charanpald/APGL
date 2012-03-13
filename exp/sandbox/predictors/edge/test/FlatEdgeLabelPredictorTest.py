
import unittest
import numpy
import logging
import sys
from apgl.predictors.edge.FlatEdgeLabelPredictor import FlatEdgeLabelPredictor 
from apgl.predictors import *
from apgl.graph import *
from apgl.kernel import *


class FlatEdgeLabelPredictorTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def testLearnModel(self):
        numVertices = 10
        numFeatures = 5

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = SparseGraph(vList, False)

        #Create a graph with 2 ego networks
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(0, 2, 0.2)
        graph.addEdge(0, 3, 0.3)

        graph.addEdge(4, 5, 0.5)
        graph.addEdge(4, 6, 0.6)
        graph.addEdge(4, 7, 0.7)

        lmbda = 0.01

        regressor = PrimalRidgeRegression(lmbda)
        logging.debug("Running prediction")
        predictor = FlatEdgeLabelPredictor(regressor)
        predictor.learnModel(graph)

        #Try using an undirected graph

        graph = SparseGraph(vList, True)

        #Create a graph with 2 ego networks
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(0, 2, 0.2)
        graph.addEdge(0, 3, 0.3)

        graph.addEdge(4, 5, 0.5)
        graph.addEdge(4, 6, 0.6)
        graph.addEdge(4, 7, 0.7)

        lmbda = 0.01

        regressor = PrimalRidgeRegression(lmbda)
        logging.debug("Running prediction")
        predictor = FlatEdgeLabelPredictor(regressor)
        predictor.learnModel(graph)

    def testPredictEdges(self):
        numVertices = 10
        numFeatures = 3

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = SparseGraph(vList, False)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(0, 4)
        graph.addEdge(0, 5)

        graph.addEdge(1, 2)
        graph.addEdge(1, 3)
        graph.addEdge(1, 4)
        graph.addEdge(1, 5)

        graph.addEdge(2, 3)
        graph.addEdge(3, 4)

        X = GraphUtils.vertexLabelPairs(graph, graph.getAllEdges())
        c = numpy.random.rand(numFeatures*2)
        logging.debug(("c=" + str(c)))

        y = numpy.dot(X, c)
        graph.addEdges(graph.getAllEdges(), y)

        lmbda = 0.001
        tol = 10**-1

        regressor = PrimalRidgeRegression(lmbda)
        predictor = FlatEdgeLabelPredictor(regressor)
        predictor.learnModel(graph)

        w = regressor.getWeights()
        logging.debug(("w=" + str(w)))
        self.assertTrue(numpy.linalg.norm(w - c) < tol)


        predY = predictor.predictEdges(graph, graph.getAllEdges())
        logging.debug((numpy.linalg.norm(predY - y)))
        self.assertTrue(numpy.linalg.norm(predY - y) < tol)

        #Now try on new edges
        testEdges = numpy.array([[2, 3], [4, 5], [6, 7]])
        testX = GraphUtils.vertexLabelPairs(graph, testEdges)

        testY = numpy.dot(testX, c)
        graph.addEdges(testEdges, testY)

        predY = predictor.predictEdges(graph, testEdges)

        logging.debug((numpy.linalg.norm(predY - testY)))
        self.assertTrue(numpy.linalg.norm(predY - testY) < 0.3)

