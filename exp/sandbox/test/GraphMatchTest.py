

import unittest
import numpy
import logging
import sys 

from apgl.graph.SparseGraph import SparseGraph 
from apgl.graph.VertexList import VertexList 
from apgl.sandbox.GraphMatch import GraphMatch 

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
        matcher = GraphMatch(algorithm="U", alpha=0.0)
        permutation, distance, time = matcher.match(self.graph1, self.graph2)

        #Checked output file - seems correct 
        
        distance2 = GraphMatch.distance(self.graph1, self.graph2, permutation)
        self.assertAlmostEquals(distance, distance2)
            
    def testDistance(self): 
        permutation = numpy.arange(self.numVertices)
        dist =  GraphMatch.distance(self.graph1, self.graph1, permutation)
        self.assertEquals(dist, 0.0)
        
        dist =  GraphMatch.distance(self.graph1, self.graph2, permutation)
        self.assertAlmostEquals(dist, 50.0)
        
        permutation = numpy.arange(self.numVertices)
        permutation[8] = 9
        permutation[9] = 8
        dist =  GraphMatch.distance(self.graph1, self.graph2, permutation)
        self.assertAlmostEquals(dist, 54.0)

if __name__ == '__main__':
    unittest.main()
