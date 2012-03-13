
import unittest
import numpy 
from apgl.sandbox.predictors.edge.RandomEdgePredictor import RandomEdgePredictor
from apgl.sandbox.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph

class  AbstractEdgePredictorTest(unittest.TestCase):
    def setUp(self):
        pass

    def testIndicesFromScores(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
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

        windowSize = 10
        predictor = RandomEdgePredictor(windowSize)
        predictor.learnModel(graph)
        scores = numpy.random.randn(numVertices)
        ind = 0 
        p, s = predictor.indicesFromScores(ind , scores)

        self.assertTrue(p.shape[0] == windowSize)
        self.assertTrue(s.shape[0] == windowSize)

        infIndices = p[numpy.nonzero(s==-float('Inf'))]

        self.assertTrue((numpy.sort(infIndices) == numpy.sort(graph.neighbours(ind))).all())


    def testCvModelSelection(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
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

        windowSize = 3
        predictor = RandomEdgePredictor(windowSize)

        folds = 5
        paramList = [[1, 2], [2, 1], [12, 1]]
        paramFunc = [predictor.setC, predictor.setD]

        errors = predictor.cvModelSelection(graph, paramList, paramFunc, folds)

        self.assertTrue(errors.shape[0] == len(paramList))

        for i in range(errors.shape[0]): 
            self.assertTrue(errors[i]>= 0 and errors[i]<= 1)
    

if __name__ == '__main__':
    unittest.main()

