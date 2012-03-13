
import unittest
import numpy 
import logging
from apgl.predictors.edge.EgoEdgePredictor import EgoEdgePredictor
from apgl.predictors import * 
from apgl.graph import *
from apgl.kernel import * 


class EgoEdgePredictorTest(unittest.TestCase):
    def setUp(self):
        pass


    def testPredictEdges(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.array([numpy.arange(numVertices)]).T)
        graph = SparseGraph(vList)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(3, 4)
        graph.addEdge(5, 6)
        graph.addEdge(4, 6)
        graph.addEdge(9, 8)
        graph.addEdge(9, 7)
        graph.addEdge(9, 6)


        vertexIndices = numpy.array([0])
        windowSize = 10
        lmbda = 1.0

        alterRegressor = PrimalRidgeRegression(lmbda)
        egoRegressor = PrimalRidgeRegression(lmbda)
        predictor = EgoEdgePredictor(alterRegressor, egoRegressor, windowSize)
        predictor.learnModel(graph)
        P, S = predictor.predictEdges(vertexIndices)

        logging.debug(P)
        logging.debug(S)

        #Test case where there is an isolate vertex
        graph.removeEdge(5, 6)
        predictor.learnModel(graph)
        P, S = predictor.predictEdges(vertexIndices)

        linearP = P

        #Test case in which we have a kernel method with linear kernel 
        kernel = LinearKernel()

        alterRegressor = PrimalRidgeRegression(lmbda)
        egoRegressor = KernelRidgeRegression(kernel, lmbda)

        tol = 10**-6
        predictor = EgoEdgePredictor(alterRegressor, egoRegressor, windowSize)
        predictor.learnModel(graph)
        P, S = predictor.predictEdges(vertexIndices)

        self.assertTrue(numpy.linalg.norm(linearP - P ) < tol )
