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
        
    def testGreedyMethod(self): 
        graph = igraph.Graph(directed=True)        
        graph.add_vertices(5)
        graph.add_edges([(0,1), (0,2), (1, 3), (2,3), (2,4)])
        graph.es["p"] = numpy.array([0, 1, 1, 0, 1])
        
        k = 3
        influenceList = MaxInfluence.celf(graph, k)
        self.assertEquals(set(influenceList), set([0,1,2]))

        #2nd example 
        graph = igraph.Graph(directed=True)        
        graph.add_vertices(8)
        graph.add_edges([(0,1), (0,2), (1, 3), (2,3), (2,4), (5,6), (6,7), (5, 4)])
        graph.es["p"] = numpy.array([0, 1, 1, 0, 1, 1, 1, 1])
        
        k = 3
        influenceList = MaxInfluence.celf(graph, k)        
        self.assertEquals(influenceList, [5, 0, 1]) 
        
        
        #Now try some random graphs 
        print("Starting random test")
        numReps = 10         
        
        for i in range(numReps): 
            n = 10 
            p = 0.1
            graph = igraph.Graph.Erdos_Renyi(n, p)
            graph.es["p"] = numpy.random.rand(graph.ecount())
            
            k = 4
            influenceList = MaxInfluence.celf(graph, k, 10000)  
            
            influenceList2 = MaxInfluence.greedyMethod(graph, k, 10000)  
            self.assertEquals(influenceList, influenceList2)

if __name__ == '__main__':
    unittest.main()

