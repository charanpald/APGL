

import unittest
import numpy
import logging
import sys 
import numpy.testing as nptst 

from apgl.graph.SparseGraph import SparseGraph 
from apgl.graph.VertexList import VertexList 
from exp.sandbox.GraphMatch import GraphMatch 

class GraphMatchTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3)
        numpy.random.seed(21)
        numpy.set_printoptions(threshold=numpy.nan, linewidth=100)
        
        #Use the example in the document
        self.numVertices = 10 
        self.numFeatures = 2 
        self.graph1 = SparseGraph(VertexList(self.numVertices, self.numFeatures))
        self.graph1.setVertices(range(self.numVertices), numpy.random.rand(self.numVertices, self.numFeatures))
        
        edges = numpy.array([[0,1], [0, 2], [0,4], [0,5], [0,8], [0,9]])
        self.graph1.addEdges(edges) 
        edges = numpy.array([[1,3], [1, 5], [1,6], [1,8], [2,9], [3,4], [3,5], [3,6], [3,7], [3,8], [3,9]])
        self.graph1.addEdges(edges)         
        edges = numpy.array([[4,2], [4, 7], [4,9], [5,8], [6, 7]])
        self.graph1.addEdges(edges)  
       
        self.graph2 = SparseGraph(VertexList(self.numVertices, self.numFeatures))
        self.graph2.setVertices(range(self.numVertices), numpy.random.rand(self.numVertices, self.numFeatures))
        
        edges = numpy.array([[0,3], [0, 4], [0,5], [0,8], [0,9], [1,2]])
        self.graph2.addEdges(edges) 
        edges = numpy.array([[1,3], [1,5], [1, 7], [1,8], [1,9], [2,3], [2,5], [3,5], [4,5], [4,6]])
        self.graph2.addEdges(edges)         
        edges = numpy.array([[4,9], [6, 8], [7,8], [7,9], [8, 9]])
        self.graph2.addEdges(edges)  

    def testMatch(self): 
        matcher = GraphMatch(algorithm="U", alpha=0.3)
        permutation, distance, time = matcher.match(self.graph1, self.graph2)

        #Checked output file - seems correct 
        
        distance2 = GraphMatch(alpha=0.0).distance(self.graph1, self.graph2, permutation)
        self.assertAlmostEquals(distance[0], distance2)
        
        #Now test case in which alpha is different 
        matcher = GraphMatch(algorithm="U", alpha=0.5)
        permutation, distance, time = matcher.match(self.graph1, self.graph2)
        distance2 = GraphMatch(alpha=0.0).distance(self.graph1, self.graph2, permutation)
        self.assertAlmostEquals(distance[0], distance2)
        
        #Test normalised distance 
        alpha = 0.0
        permutation, distance, time = GraphMatch(algorithm="U", alpha=alpha).match(self.graph1, self.graph2)
        distance2 = GraphMatch(alpha=alpha).distance(self.graph1, self.graph2, permutation, True)
        self.assertAlmostEquals(distance[1], distance2)
        
        alpha = 1.0
        permutation, distance, time = GraphMatch(algorithm="U", alpha=alpha).match(self.graph1, self.graph2)
        distance2 = GraphMatch(alpha=alpha).distance(self.graph1, self.graph2, permutation, True)
        self.assertAlmostEquals(distance[1], distance2, 5)
            
    def testDistance(self): 
        permutation = numpy.arange(self.numVertices)
        dist =  GraphMatch(alpha=0.0).distance(self.graph1, self.graph1, permutation)
        self.assertEquals(dist, 0.0)
        
        dist =  GraphMatch(alpha=0.0).distance(self.graph1, self.graph2, permutation)
        self.assertAlmostEquals(dist, 50.0)
        
        permutation = numpy.arange(self.numVertices)
        permutation[8] = 9
        permutation[9] = 8
        dist =  GraphMatch(alpha=0.0).distance(self.graph1, self.graph2, permutation)
        self.assertAlmostEquals(dist, 54.0)
        
        #Try graphs of unequal size 
        graph3 = self.graph1.subgraph(range(8))
        permutation = numpy.arange(self.numVertices)
        dist1 =  GraphMatch(alpha=0.0).distance(self.graph1, graph3, permutation)
        dist1a =  GraphMatch(alpha=0.0).distance(graph3, self.graph1, permutation)
        self.assertEquals(dist1, dist1a)

        graph3 = self.graph1.subgraph(range(5))
        dist2 =  GraphMatch(alpha=0.0).distance(self.graph1, graph3, permutation)
        dist2a =  GraphMatch(alpha=0.0).distance(graph3, self.graph1, permutation)
        self.assertEquals(dist2, dist2a)
        self.assertTrue(dist1 < dist2)
        
        #Test case where alpha!=0 
        alpha = 1.0
        permutation = numpy.arange(self.numVertices)
        distance = GraphMatch(alpha=alpha).distance(self.graph1, self.graph2, permutation, False)
        C = GraphMatch(alpha=alpha).vertexSimilarities(self.graph1, self.graph2)
        distance2 = -numpy.trace(C)
        self.assertEquals(distance, distance2)
        
        #Check case where we want non negativve distance even when alpha!=0 
        distance = GraphMatch(alpha=alpha).distance(self.graph1, self.graph2, permutation, True, True)
        self.assertTrue(distance >= 0)
        
        permutation = numpy.arange(self.numVertices)
        distance = GraphMatch(alpha=alpha).distance(self.graph1, self.graph1, permutation, True, True)
        self.assertEquals(distance, 0)
        
    def testDistance2(self): 
        permutation = numpy.arange(self.numVertices)
        dist =  GraphMatch(alpha=0.0).distance2(self.graph1, self.graph1, permutation)
        self.assertEquals(dist, 0.0)
        
        dist =  GraphMatch(alpha=0.0).distance2(self.graph1, self.graph2, permutation)
        dist2 = GraphMatch(alpha=0.0).distance(self.graph1, self.graph2, permutation, True)
        self.assertAlmostEquals(dist, dist2)
        
        permutation = numpy.arange(self.numVertices)
        permutation[8] = 9
        permutation[9] = 8
        dist =  GraphMatch(alpha=0.0).distance2(self.graph1, self.graph2, permutation)
        dist2 = GraphMatch(alpha=0.0).distance(self.graph1, self.graph2, permutation, True)
        self.assertAlmostEquals(dist, dist2)
        
        #Try graphs of unequal size 
        graph3 = self.graph1.subgraph(range(8))
        permutation = numpy.arange(self.numVertices)
        dist1 =  GraphMatch(alpha=0.0).distance2(self.graph1, graph3, permutation)
        dist1a =  GraphMatch(alpha=0.0).distance2(graph3, self.graph1, permutation)
        self.assertEquals(dist1, dist1a)

        graph3 = self.graph1.subgraph(range(5))
        dist2 =  GraphMatch(alpha=0.0).distance2(self.graph1, graph3, permutation)
        dist2a =  GraphMatch(alpha=0.0).distance2(graph3, self.graph1, permutation)
        self.assertEquals(dist2, dist2a)
        self.assertTrue(dist1 < dist2)
        
        #Test case where alpha!=0 
        alpha = 1.0
        permutation = numpy.arange(self.numVertices)
        distance = GraphMatch(alpha=alpha).distance2(self.graph1, self.graph1, permutation)
        self.assertEquals(distance, 0.0)
        
        #Check distances are between 0 and 1 
        for i in range(100): 
            alpha = numpy.random.rand()
            permutation = numpy.random.permutation(self.numVertices)
            
            distance = GraphMatch(alpha=alpha).distance2(self.graph1, self.graph1, permutation)
            self.assertTrue(0<=distance<=1)
    
    def testVertexSimilarities(self): 
        matcher = GraphMatch(alpha=0.0)
        C = matcher.vertexSimilarities(self.graph1, self.graph1) 
        
        C = matcher.vertexSimilarities(self.graph2, self.graph2) 
        
        Cdiag = numpy.diag(C)
        nptst.assert_array_almost_equal(Cdiag, 2*numpy.ones(Cdiag.shape[0]))
        
        #Now compute trace(C)/||C||
        #print(numpy.trace(C)/numpy.linalg.norm(C))
        
if __name__ == '__main__':
    unittest.main()
