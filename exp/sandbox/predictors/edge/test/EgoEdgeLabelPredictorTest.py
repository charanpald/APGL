
import unittest
import numpy
import logging
import sys
import random 
from apgl.predictors.edge.EgoEdgeLabelPredictor import EgoEdgeLabelPredictor
from apgl.predictors import *
from apgl.graph import *
from apgl.kernel import *
from apgl.util import *
from apgl.generator import * 


class EgoEdgeLabelPredictorTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def testLearnModel(self):
        numVertices = 10
        numFeatures = 5

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = SparseGraph(vList, False)

        #Create a graph with 2 ego networks
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)

        graph.addEdge(4, 5)
        graph.addEdge(4, 6)
        graph.addEdge(4, 7)

        lmbda = 0.01

        alterRegressor = PrimalRidgeRegression(lmbda)
        egoRegressor = PrimalRidgeRegression(lmbda)
        #logging.debug("Running prediction")
        predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)
        predictor.learnModel(graph)

    def testPredictEdges(self):
        random.seed(21)
        numpy.random.seed(21)

        numVertices = 10
        numFeatures = 5

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = SparseGraph(vList, False)

        #Ego coefficients are just values of the egos
        numEgos = 3 
        C = vList.getVertices(list(range(0, numEgos)))

        for i in range(numEgos):
            alterVertices = numpy.arange(0, numVertices)
            alterVertices = numpy.setdiff1d(alterVertices, numpy.array([i]))
            X = vList.getVertices(alterVertices)
            y = numpy.dot(X, C[i, :].T)

            for j in range(alterVertices.shape[0]):
                graph.addEdge(i, alterVertices[j], y[j])

        lmbda = 0.00001

        alterRegressor = PrimalRidgeRegression(lmbda)
        egoRegressor = PrimalRidgeRegression(lmbda)
        predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)
        predictor.learnModel(graph)

        edges = graph.getAllEdges()
        predY = predictor.predictEdges(graph, edges)
        y = numpy.zeros(edges.shape[0])

        for i in range(edges.shape[0]):
            y[i] = graph.getEdge(edges[i ,0], edges[i, 1])

        tol = 10**-1

        self.assertTrue(numpy.linalg.norm(predY - y) < tol)

        #Now try on a new edge
        alterVertices = numpy.arange(0, numVertices)
        alterVertices = numpy.setdiff1d(alterVertices, numpy.array([numEgos]))

        edges = numpy.ones((alterVertices.shape[0], 2), numpy.int)*numEgos
        edges[:, 1] = alterVertices

        predY = predictor.predictEdges(graph, edges)
        y = numpy.zeros(edges.shape[0])


        X = vList.getVertices(alterVertices)
        y = numpy.dot(X, vList.getVertex(numEgos))


        logging.debug(y)
        logging.debug(predY)
        error= Evaluator.rootMeanSqError(y, predY)
        logging.debug(error)
        self.assertTrue(error < 0.4)

    #Test the stability of the network with small numbers of egos in each network 
    def testPredictEdges2(self):
        numVertices = 100
        numFeatures = 5
        
        #Elements of X are in range [-1, 1]
        verticies = (numpy.random.rand(numVertices, numFeatures) - 0.5)*2
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(verticies)

        p = 0.2
        graph = SparseGraph(vList, False)
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

        #Now extract all pairs of vertices and assign values to edges
        allEdges = graph.getAllEdges()
        logging.debug("Number of edges: " + str(allEdges.shape[0]))

        egos = vList.getVertices(allEdges[:, 0])
        alters = vList.getVertices(allEdges[:, 1])

        C = numpy.random.rand(numFeatures, numFeatures)
        alterCs = numpy.dot(egos, C)
        
        WReal = numpy.dot(vList.getVertices(list(range(numVertices))), C)

        noise = 0.1
        yReal = numpy.sum(alters * alterCs, 1)
        yReal = yReal + noise * numpy.random.rand(allEdges.shape[0])
        
        #yReal = (yReal - numpy.min(yReal)) /(numpy.max(yReal) - numpy.min(yReal))

        graph.addEdges(allEdges, yReal)

        #logging.debug("yReal.shape=" + str(yReal.shape))
        #logging.debug("yReal = " + str(yReal))

        #Question: Can we recover C by learning using the examples?
        #Answer: yes
        lmbda1 = 0.01
        lmbda2 = 0.01
        alterRegressor = PrimalRidgeRegression(lmbda1)
        egoRegressor = PrimalRidgeRegression(lmbda2)
        edgeLearner = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

        W = edgeLearner.learnModel(graph)
        C2 = egoRegressor.getWeights()

        predY = edgeLearner.predictEdges(graph, graph.getAllEdges())

        error= Evaluator.rootMeanSqError(yReal, predY)

        logging.debug(("error = " +  str(error)))
        self.assertTrue(error < 10**-1)
        #self.assertTrue(numpy.linalg.norm(C - C2) < 10**-1)