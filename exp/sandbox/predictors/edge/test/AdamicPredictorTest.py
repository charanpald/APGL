
import unittest
import numpy
from apgl.predictors.edge.AdamicPredictor import AdamicPredictor
from apgl.graph import *


class  AdamicPredictorTest(unittest.TestCase):


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

        windowSize = 9
        predictor = AdamicPredictor(windowSize)
        predictor.learnModel(graph)
        P, S = predictor.predictEdges(numpy.array([5]))
        self.assertTrue((P[0, :] == numpy.array([9, 5, 4, 8, 7, 3, 2, 1, 0])).all())

        P, S = predictor.predictEdges(numpy.array([6]))
        self.assertTrue((P[0, :] == numpy.array([6, 3, 8, 7, 2, 1, 0, 9, 5])).all())

        P, S = predictor.predictEdges(numpy.array([5, 6]))
        self.assertTrue((P[0, :] == numpy.array([ 9, 5, 4, 8, 7, 3, 2, 1, 0])).all())
        self.assertTrue((P[1, :] == numpy.array([ 6, 3, 8, 7, 2, 1, 0, 9, 5])).all())

        windowSize = 5
        predictor.setWindowSize(windowSize)
        P, S = predictor.predictEdges(numpy.array([5, 6]))
        self.assertTrue((P[0, :] == numpy.array([ 9, 5, 4, 8, 7])).all())
        self.assertTrue((P[1, :] == numpy.array([ 6, 3, 8, 7, 2])).all())

if __name__ == '__main__':
    unittest.main()