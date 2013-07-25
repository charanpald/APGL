import numpy 
import unittest
import logging
import ctypes
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
import numpy.testing as nptst 

from exp.influence2.MaxInfluence import MaxInfluence 

class  MaxInfluenceTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
        numpy.set_printoptions(suppress=True, precision=3)
    
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
        numVertices = 5        
        graph.add_vertices(numVertices)
        
        
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
    
    def testSimulateCascades(self): 
        #Test submodularity 
        n = 20 
        p = 0.1
        graph = igraph.Graph.Erdos_Renyi(n, p)
        graph.es["p"] = numpy.array(numpy.random.rand(graph.ecount()) < 0.5, numpy.bool)
        
        numRuns = 1
        influences = numpy.zeros(n)
        
        activeVertexInds = set([0])  
        lastInfluence = MaxInfluence.simulateCascades(graph, activeVertexInds, numRuns)
        
        for i in range(n): 
            activeVertexInds = set([0, i])  
            influences[i] = MaxInfluence.simulateCascades(graph, activeVertexInds, numRuns)

        marginalGains = (influences - lastInfluence)
        
        activeVertexInds = set([0, 1]) 
        lastInfluence = MaxInfluence.simulateCascades(graph, activeVertexInds, numRuns)
        
        for i in range(n): 
            activeVertexInds = set([0, 1, i])  
            influences[i] = MaxInfluence.simulateCascades(graph, activeVertexInds, numRuns)
            

        marginalGains2 = (influences - lastInfluence)
        
        self.assertTrue((marginalGains2 <= marginalGains).all())
        
    #@unittest.skip("")
    def testCelf(self): 
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
        #self.assertEquals(influenceList, [5, 0, 1]) 
        
        #Now try some random graphs 
        numReps = 10      
        numRuns = 1
        
        for i in range(numReps): 
            print(i)
            n = numpy.random.randint(10, 50) 
            p = numpy.random.rand()
            graph = igraph.Graph.Erdos_Renyi(n, p)
            
            k = numpy.random.randint(5, 10)
            influenceList = MaxInfluence.celf(graph, k, numRuns, p=1)  
            
            influenceList2 = MaxInfluence.greedyMethod(graph, k, numRuns, p=1)  
            self.assertEquals(influenceList, influenceList2)

    def testSimulateInitialCascade(self): 
        n = 100 
        p = 0.01
        graph = igraph.Graph.Erdos_Renyi(n, p)
        
        numRuns = 10000 
                
        influences = numpy.zeros(n)
        
        for i in range(numRuns): 
            influences += MaxInfluence.simulateInitialCascade(graph, p=0.2)
        
        influences /= numRuns         
        
        
        #Now compute influence via the other way 
        influences2 = numpy.zeros(n)
        
        for i in range(n): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([i]), numRuns, p=0.2)
            
        nptst.assert_array_almost_equal(influences, influences2, 1)

if __name__ == '__main__':
    unittest.main()

