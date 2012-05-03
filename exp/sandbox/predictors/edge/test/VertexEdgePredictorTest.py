
import unittest
import numpy
import logging
import sys 
from exp.sandbox.predictors.edge.VertexEdgePredictor import VertexEdgePredictor
from apgl.predictors.KernelRidgeRegression import KernelRidgeRegression
from apgl.kernel.LinearKernel import LinearKernel 
from apgl.graph import *

class  VertexEdgePredictorTest(unittest.TestCase):
    def setUp(self):
        pass

    def testPredictEdge(self):
        """
        Test edge prediction on a very simple example where vertices have a scalar
        label. 
        """
        numVertices = 10
        numFeatures = 1

        numpy.random.seed(21)

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.randn(numVertices, numFeatures))

        logging.debug((vList.getVertices(list(range(0, numVertices)))))

        #Now setup the data so there is a set of coefficient over vertices
        #In this case, if two vertices sum to positive values there is an edge 
        c = numpy.array([1, 1])
        X = numpy.zeros((numVertices**2, numFeatures*2))
        y = numpy.zeros(numVertices**2)

        graph = SparseGraph(vList)

        for i in range(numVertices):
            for j in range(numVertices):
                ind = i*numVertices + j 
                X[ind, 0:numFeatures] = vList.getVertex(i)
                X[ind, numFeatures:numFeatures*2] = vList.getVertex(j)
                
                y[ind] = numpy.dot(X[ind, :], c)

                if y[ind] > 0:
                    graph.addEdge(i, j)

        logging.debug((graph.getNumEdges()))
        logging.debug((graph.getAllEdges()))

        trainGraph = SparseGraph(vList)
        allEdges = graph.getAllEdges()

        perm = numpy.random.permutation(allEdges.shape[0])[0:10]
        trainEdges = allEdges[perm, :]
        trainGraph.addEdges(trainEdges)

        logging.debug((trainGraph.getAllEdges()))        

        kernel = LinearKernel()
        lmbda = 0.1

        krr = KernelRidgeRegression(kernel, lmbda)

        windowSize = 3
        predictor = VertexEdgePredictor(krr)
        P, S = predictor.predictEdges(trainGraph, numpy.arange(numVertices), windowSize)

        logging.debug(P)
        logging.debug(S)

        #We should get a low error
        testEdges = allEdges[numpy.setdiff1d(numpy.arange(graph.getNumEdges()), perm), :]
        error = 0.0

        logging.debug(testEdges)

        for i in range(testEdges.shape[0]):

            if testEdges[i, 1] not in P[testEdges[i, :], :]:
                error = 1.0/testEdges.shape[0]

        logging.debug(error)

        self.assertEquals(error, 0.0)

    def testPredictEdge(self):
        """
        Test edge prediction on a more complex example.
        """
        numVertices = 10
        numFeatures = 5

        numpy.random.seed(21)

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.randn(numVertices, numFeatures))

        logging.debug((vList.getVertices(list(range(0, numVertices)))))

        #Now setup the data so there is a set of coefficient over vertices
        #In this case, if two vertices sum to positive values there is an edge
        c = numpy.random.randn(numFeatures)
        c = numpy.r_[c, c]
        X = numpy.zeros((numVertices**2, numFeatures*2))
        y = numpy.zeros(numVertices**2)

        graph = SparseGraph(vList)

        for i in range(numVertices):
            for j in range(numVertices):
                ind = i*numVertices + j
                X[ind, 0:numFeatures] = vList.getVertex(i)
                X[ind, numFeatures:numFeatures*2] = vList.getVertex(j)

                y[ind] = numpy.dot(X[ind, :], c)

                if y[ind] > 0:
                    graph.addEdge(i, j)

        logging.debug((graph.getNumEdges()))
        logging.debug((graph.getAllEdges()))

        trainGraph = SparseGraph(vList)
        allEdges = graph.getAllEdges()

        perm = numpy.random.permutation(allEdges.shape[0])[0:10]
        trainEdges = allEdges[perm, :]
        trainGraph.addEdges(trainEdges)

        logging.debug((trainGraph.getAllEdges()))

        kernel = LinearKernel()
        lmbda = 0.1

        krr = KernelRidgeRegression(kernel, lmbda)

        windowSize = 3
        predictor = VertexEdgePredictor(krr, windowSize)
        predictor.learnModel(graph)
        P, S = predictor.predictEdges(numpy.arange(numVertices))

        logging.debug(P)
        logging.debug(S)

        #We should get a low error
        testEdges = allEdges[numpy.setdiff1d(numpy.arange(graph.getNumEdges()), perm), :]
        error = 0.0

        logging.debug(testEdges)

        for i in range(testEdges.shape[0]):

            if testEdges[i, 1] not in P[testEdges[i, :], :]:
                error = 1.0/testEdges.shape[0]

        logging.debug(error)

        self.assertTrue(error <= 0.1)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    unittest.main()
