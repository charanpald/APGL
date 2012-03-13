
import unittest
import numpy
from apgl.predictors.edge.RandomEdgePredictor import RandomEdgePredictor
from apgl.graph import *


class  RandomEdgePredictorTest(unittest.TestCase):


    def testPredictEdge(self):
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
        vertexIndices = numpy.array([4,3, 1, 6, 5])
        predictor.learnModel(graph)
        P, S = predictor.predictEdges(vertexIndices)

        self.assertEquals(P.shape[0], 5)
        self.assertEquals(P.shape[1], windowSize)

        for i in range(vertexIndices.shape[0]):
            infIndices = P[i, numpy.nonzero(S[i, :]==-float('Inf'))]
            self.assertTrue((numpy.sort(graph.neighbours(vertexIndices[i])) == numpy.sort(infIndices)).all())

if __name__ == '__main__':
    unittest.main()