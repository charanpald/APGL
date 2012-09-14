

import unittest
import numpy
import logging
import sys 
import numpy.testing as nptst 

from apgl.generator import SmallWorldGenerator 
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
        
        #Test empty graph
        alpha = 0.0
        graph1 = SparseGraph(VertexList(0, 0))
        graph2 = SparseGraph(VertexList(0, 0))
        
        permutation, distance, time = GraphMatch(algorithm="U", alpha=alpha).match(graph1, graph2)
        
        nptst.assert_array_equal(permutation, numpy.array([], numpy.int))
        self.assertEquals(distance, [0, 0, 0])
        
        #Test where 1 graph is empty 
        permutation, distance, time = GraphMatch(algorithm="U", alpha=alpha).match(graph1, self.graph1)
        self.assertEquals(numpy.linalg.norm(self.graph1.getWeightMatrix())**2, distance[0])
        self.assertEquals(distance[1], 1)
        self.assertEquals(distance[2], 1)
        
        permutation, distance, time = GraphMatch(algorithm="U", alpha=alpha).match(self.graph1, graph1)
        self.assertEquals(numpy.linalg.norm(self.graph1.getWeightMatrix())**2, distance[0])
        self.assertEquals(distance[1], 1)
        self.assertEquals(distance[2], 1)
        
        alpha = 1.0
        permutation, distance, time = GraphMatch(algorithm="U", alpha=alpha).match(graph1, self.graph1)
        self.assertEquals(numpy.linalg.norm(self.graph1.getWeightMatrix())**2, distance[0])
        
        V2 = self.graph1.vlist.getVertices()
        V1 = numpy.zeros(V2.shape)
        C = GraphMatch(algorithm="U", alpha=alpha).matrixSimilarity(V1, V2)
        dist = numpy.trace(C)/numpy.linalg.norm(C)
        
        self.assertAlmostEquals(distance[1], -dist, 4)
        self.assertAlmostEquals(distance[2], -dist, 4)
        
        permutation, distance, time = GraphMatch(algorithm="U", alpha=alpha).match(self.graph1, graph1)
        self.assertEquals(numpy.linalg.norm(self.graph1.getWeightMatrix())**2, distance[0])
        self.assertAlmostEquals(distance[1], -dist, 4)
        self.assertAlmostEquals(distance[2], -dist, 4)
        
        #Test one graph which is a subgraph of another 
        p = 0.2 
        k = 10 
        numVertices = 20
        generator = SmallWorldGenerator(p, k)
        graph = SparseGraph(VertexList(numVertices, 2))
        graph = generator.generate(graph)
        
        subgraphInds = numpy.random.permutation(numVertices)[0:10]
        subgraph = graph.subgraph(subgraphInds)
        
        matcher = GraphMatch(algorithm="U", alpha=0.0)
        permutation, distance, time = matcher.match(graph, subgraph)
        distance = matcher.distance(graph, subgraph, permutation, True, True)
        
        self.assertTrue(distance < 1)
        
            
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
        
        #Check case where both graphs are empty 
        graph1 = SparseGraph(VertexList(0, 0))
        graph2 = SparseGraph(VertexList(0, 0))
        
        permutation = numpy.array([], numpy.int)
        distance = GraphMatch(alpha=alpha).distance(graph1, graph1, permutation, True, True)
        self.assertEquals(distance, 0)
        
        #Now, just one graph is empty 
        #Distance is always 1 due to normalisations 
        alpha = 0.0
        permutation = numpy.arange(10, dtype=numpy.int)
        distance = GraphMatch(alpha=alpha).distance(self.graph1, graph1, permutation, True, True)
        self.assertEquals(distance, 1.0)
        
        permutation = numpy.arange(10, dtype=numpy.int)
        distance = GraphMatch(alpha=alpha).distance(self.graph2, graph1, permutation, True, True)
        self.assertEquals(distance, 1.0)
        
        #distance = GraphMatch(alpha=alpha).distance(self.graph1, graph1, permutation, False, False)
        #self.assertEquals(distance, numpy.linalg.norm(self.graph1.getWeightMatrix())**2)
        
        alpha = 0.9 
        matcher = GraphMatch("U", alpha=alpha)
        permutation, distanceVector, time = matcher.match(self.graph2, graph1)
        distance = matcher.distance(self.graph2, graph1, permutation, True, True)
        self.assertEquals(distance, 1.0)        
        
        alpha = 1.0
        permutation = numpy.arange(10, dtype=numpy.int)
        distance = GraphMatch(alpha=alpha).distance(self.graph1, graph1, permutation, True, True)
        self.assertEquals(distance, 1.0)
        
        permutation = numpy.arange(10, dtype=numpy.int)
        distance = GraphMatch(alpha=alpha).distance(self.graph2, graph1, permutation, True, True)
        self.assertEquals(distance, 1.0)
        
        alpha = 0.5
        permutation = numpy.arange(10, dtype=numpy.int)
        distance = GraphMatch(alpha=alpha).distance(self.graph2, graph1, permutation, True, True)
        self.assertEquals(distance, 1.0)
           
        #Test on unequal graphs and compare against distance from graphm 
        alpha = 0.5 
        matcher = GraphMatch(alpha=alpha)
        permutation, distanceVector, time = matcher.match(self.graph1, self.graph2)
        distance = matcher.distance(self.graph1, self.graph2, permutation, True, False)
        
        self.assertAlmostEquals(distanceVector[1], distance, 3)
        
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
        
        Cdiag = numpy.diag(C)
        nptst.assert_array_almost_equal(Cdiag, numpy.ones(Cdiag.shape[0]))
        
        #Now compute trace(C)/||C||
        #print(numpy.trace(C)/numpy.linalg.norm(C))
        
        #Test use of feature inds 
        matcher = GraphMatch(alpha=0.0, featureInds=numpy.array([0]))
        
        C = matcher.vertexSimilarities(self.graph1, self.graph2) 
        
        #Now, let's vary the non-used feature 
        self.graph1.vlist[:, 1] = 0
        C2 = matcher.vertexSimilarities(self.graph1, self.graph2) 
        nptst.assert_array_equal(C, C2)
        
        self.graph2.vlist[:, 1] = 0
        C2 = matcher.vertexSimilarities(self.graph1, self.graph2) 
        nptst.assert_array_equal(C, C2)
        
        #Vary used feature 
        self.graph1.vlist[:, 0] = 0
        C2 = matcher.vertexSimilarities(self.graph1, self.graph2) 
        self.assertTrue((C != C2).any())
  
    def testMatrixSimilarity(self):
        numExamples = 5 
        numFeatures = 3 
        V1 = numpy.random.rand(numExamples, numFeatures)
          
        matcher = GraphMatch(alpha=0.0)
        C = matcher.matrixSimilarity(V1, V1)
        Cdiag = numpy.diag(C)
        nptst.assert_array_almost_equal(Cdiag, numpy.ones(Cdiag.shape[0]))
        
        V1[:, 2] *= 10 
        C2 = matcher.matrixSimilarity(V1, V1)
        Cdiag = numpy.diag(C2)
        nptst.assert_array_almost_equal(Cdiag, numpy.ones(Cdiag.shape[0]))      
        nptst.assert_array_almost_equal(C, C2)
        
        #print("Running match")
        J = numpy.ones((numExamples, numFeatures))
        Z = numpy.zeros((numExamples, numFeatures))

        C2 = matcher.matrixSimilarity(J, Z)
        #This should be 1 ideally 
        
        
        nptst.assert_array_almost_equal(C2, numpy.ones(C2.shape))  
        
        C2 = matcher.matrixSimilarity(J, J)
        nptst.assert_array_almost_equal(C2, numpy.ones(C2.shape))  
        
      
if __name__ == '__main__':
    unittest.main()
