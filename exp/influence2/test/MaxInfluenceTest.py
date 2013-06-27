import numpy 
import unittest
import logging
import ctypes
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 

from exp.influence2.MaxInfluence import MaxInfluence 

class  MaxInfluenceTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
    
    
    def testGreedyMethod(self): 
        graph = igraph.Graph(directed=True)        
        graph.add_vertices(5)
        
        graph.add_edges([(0,1), (0,2), (1, 3), (2,3), (2,4)])
        graph.es["p"] = numpy.array([0, 1, 1, 0, 1])
        
        k = 3
        influenceList = MaxInfluence.greedyMethod(graph, k)
        
        self.assertEquals(set(influenceList), set([0,1,2]))
    
    def testSimulateCascade(self): 
        graph = igraph.Graph(directed=True)        
        graph.add_vertices(5)
        
        
        graph.add_edges([(0,1), (0,2), (1, 3), (2,3), (2,4)])
        graph.es["p"] = numpy.array([0, 1, 1, 0, 1])

        activeVertexInds = set([0])        
        outputInds = MaxInfluence.simulateCascade(graph, activeVertexInds)
        self.assertEquals(outputInds, set([0,2,4])) 
        
        graph.es["p"] = numpy.array([0, 1, 0, 0, 1])
        outputInds = MaxInfluence.simulateCascade(graph, activeVertexInds)
        self.assertEquals(outputInds, set([0, 2, 4])) 
        
        graph.es["p"] = numpy.array([0, 0, 0, 0, 0])
        outputInds = MaxInfluence.simulateCascade(graph, activeVertexInds)
        self.assertEquals(outputInds, set([0])) 
        

if __name__ == '__main__':
    unittest.main()

