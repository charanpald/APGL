
import logging
import unittest
import numpy 
from apgl.predictors.edge.EdgePredictor import EdgePredictor
from apgl.graph import *

class  EdgePredictorTest(unittest.TestCase):
    def testComputeC(self):
        numExamples = 11
        numFeatures = 1 
        
        vList = VertexList(numExamples, numFeatures)
        graph = SparseGraph(vList)

        #Try the graph in "A regularisation framework for learning from graph data"
        clique1 = [0,1,2,3,4]
        clique2 = [6,7,8,9,10]

        for i in clique1:
            for j in clique1:
                if i!=j:
                    graph.addEdge(i, j)

        for i in clique2:
            for j in clique2:
                if i!=j:
                    graph.addEdge(i, j)

        graph.addEdge(2, 5)
        graph.addEdge(5, 6)
        graph.addEdge(2, 6)

        #logging.debug(graph.getAllEdges())

        alpha = 0.9

        edgePredictor = EdgePredictor()

        C = edgePredictor.computeC(graph, alpha)

        #Test the values from the relative ranking section 
        self.assertAlmostEqual(C[6, 2], 0.99, 2)
        self.assertAlmostEqual(C[6, 5], 0.87, 2)
        self.assertAlmostEqual(C[6, 7], 1.33, 2)
        self.assertAlmostEqual(C[6, 1], 0.56, 2)

        #Test the values for the 1st link prediction table
        graph.removeEdge(2, 6)
        C = edgePredictor.computeC(graph, alpha)
        self.assertAlmostEqual(C[6, 2], 0.48, 2)
        self.assertAlmostEqual(C[6, 1], 0.29, 2)
        self.assertAlmostEqual(C[2, 7], 0.29, 2)
        self.assertAlmostEqual(C[7, 1], 0.18, 2)
        self.assertAlmostEqual(C[5, 1], 0.52, 2)

        #Now the 2nd 
        alpha = 0.95 
        C = edgePredictor.computeC(graph, alpha)
        self.assertAlmostEqual(C[6, 2], 1.27, 2)
        self.assertAlmostEqual(C[6, 1], 0.93, 2)
        self.assertAlmostEqual(C[2, 7], 0.93, 2)
        self.assertAlmostEqual(C[7, 1], 0.69, 2)
        self.assertAlmostEqual(C[5, 1], 1.16, 2)


        edge = edgePredictor.predictEdge(graph, alpha)
        edge = list(edge)
        edge.sort()
        self.assertEquals(edge, [2,6])

        logging.debug(edge)

if __name__ == '__main__':
    unittest.main()

