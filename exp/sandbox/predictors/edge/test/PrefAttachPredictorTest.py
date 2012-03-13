

import unittest
import numpy 
from apgl.predictors.edge.PrefAttachPredictor import PrefAttachPredictor
from apgl.graph import * 

class  PrefAttachPredictorTest(unittest.TestCase):


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
        predictor = PrefAttachPredictor(windowSize)
        predictor.learnModel(graph)
        P, S = predictor.predictEdges(numpy.array([5]))
        

        self.assertTrue((P[0, :] == numpy.array([9,0,4,3,2,1,8,7,5])).all())

        windowSize = 5
        predictor.setWindowSize(windowSize)
        P, S = predictor.predictEdges(numpy.array([5]))
        
        self.assertTrue((P[0, :] == numpy.array([9,0,4,3,2])).all())

        windowSize = 9
        predictor.setWindowSize(windowSize)
        P, S = predictor.predictEdges(numpy.array([0,1,2]))
        self.assertTrue((P[0, :] == numpy.array([9,6,0, 4,8,7,5,3,2])).all())
        self.assertTrue((P[1, :] == numpy.array([9,6,4,3,1,8,7,5,2])).all())
        self.assertTrue((P[2, :] == numpy.array([9,6,4,3,2,8,7,5,1])).all())

if __name__ == '__main__':
    unittest.main()

