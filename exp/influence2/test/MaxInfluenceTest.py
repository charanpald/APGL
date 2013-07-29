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
        
        k = 3
        influenceList = MaxInfluence.greedyMethod(graph, k)
        
        self.assertEquals(len(influenceList), k)
  
    def testGreedyMethod2(self): 
        graph = igraph.Graph()        
        graph.add_vertices(5)
        
        graph.add_edges([(0,1), (0,2), (1, 3), (2,3), (2,4)])
        
        k = 3
        influenceList = MaxInfluence.greedyMethod2(graph, k)
        
        self.assertEquals(len(influenceList), k)
        
        
        #Now try some random graphs 
        numReps = 10      
        numRuns = 1
        
        for i in range(numReps): 
            n = numpy.random.randint(10, 50) 
            p = numpy.random.rand()
            graph = igraph.Graph.Erdos_Renyi(n, p)
            
            k = numpy.random.randint(5, 10)
            influenceList = MaxInfluence.greedyMethod2(graph, k, numRuns, p=1)  
            
            influenceList2 = MaxInfluence.greedyMethod(graph, k, numRuns, p=1)  
            self.assertEquals(influenceList, influenceList2)
  
        #Test with p!=1
        graph = igraph.Graph()        
        graph.add_vertices(8)
        graph.add_edges([(0,1), (0,2), (0, 3), (0, 4), (1,6), (2,5), (5,7)])
        
        k = 7
        influenceList, influenceScores = MaxInfluence.greedyMethod2(graph, k, numRuns=1000, p=0.1, verbose=True)  
        influenceList2, influenceScores2 = MaxInfluence.greedyMethod(graph, k, numRuns=1000, verbose=True, p=0.1)
            
        self.assertEquals(influenceList[0:3], influenceList2[0:3])  
  
    def testSimulateCascade(self): 
        graph = igraph.Graph(directed=True)
        numVertices = 5        
        graph.add_vertices(numVertices)
        
        
        graph.add_edges([(0,2), (1, 3), (2,4)])

        activeVertexInds = set([0])        
        outputInds = MaxInfluence.simulateCascade(graph, activeVertexInds, p=1)
        self.assertEquals(outputInds, set([0,2,4])) 
        
        outputInds = MaxInfluence.simulateCascade(graph, activeVertexInds, p=1)
        self.assertEquals(outputInds, set([0, 2, 4])) 
        
        graph = igraph.Graph(directed=True)
        numVertices = 5        
        graph.add_vertices(numVertices)
        outputInds = MaxInfluence.simulateCascade(graph, activeVertexInds, p=1)
        self.assertEquals(outputInds, set([0])) 
    
    def testSimulateCascades(self): 
        #Test submodularity 
        n = 20 
        p = 0.1
        graph = igraph.Graph.Erdos_Renyi(n, p)
        
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
        
        #self.assertTrue((marginalGains2 <= marginalGains).all())
        
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
            
        #Test with p!=1
        graph = igraph.Graph()        
        graph.add_vertices(8)
        graph.add_edges([(0,1), (0,2), (0, 3), (0, 4), (1,6), (2,5), (5,7)])
        
        k = 7
        influenceList, influenceScores = MaxInfluence.celf(graph, k, numRuns=1000, p=0.1, verbose=True)  
        influenceList2, influenceScores2 = MaxInfluence.greedyMethod(graph, k, numRuns=1000, verbose=True, p=0.1)

        self.assertEquals(influenceList[0:3], influenceList2[0:3])

    @unittest.skip("")
    def testSimulateAllCascades(self): 
        n = 100 
        p = 0.01
        graph = igraph.Graph.Erdos_Renyi(n, p)
        
        numRuns = 10000 
                
        influences = numpy.zeros(n)
        for i in range(numRuns): 
            influences += MaxInfluence.simulateAllCascades(graph, [], p=0.2)
        influences /= numRuns         
        
        #Now compute influence via the other way 
        influences2 = numpy.zeros(n)
        for i in range(n): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([i]), numRuns, p=0.2)
        nptst.assert_array_almost_equal(influences, influences2, 1)
        
        #Now test with p=1
        influences = MaxInfluence.simulateAllCascades(graph, [], p=1.0)
        
        influences2 = numpy.zeros(n)
        for i in range(n): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([i]), 1, p=1.0)
            
        nptst.assert_array_almost_equal(influences, influences2)
        
        #Test where we have an active vertex
        influences = MaxInfluence.simulateAllCascades(graph, [0, 10, 20], p=1.0)
        
        influences2 = numpy.zeros(n)
        for i in range(n): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([0, 10, 20, i]), 1, p=1.0)
            
        nptst.assert_array_almost_equal(influences, influences2)
        
        #Now test with p=0
        influences = MaxInfluence.simulateAllCascades(graph, [], p=0.0)
        
        influences2 = numpy.zeros(n)
        for i in range(n): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([i]), 1, p=0.0)
            
        nptst.assert_array_almost_equal(influences, influences2)
        
        #See if we get more accurate results on a small graph 
        numRuns = 100000 
        graph = igraph.Graph()        
        graph.add_vertices(8)
        graph.add_edges([(0,1), (0,2), (1, 3), (2,3), (2,4), (5,6), (6,7), (5, 4)])
        
        influences = numpy.zeros(graph.vcount())
        for i in range(numRuns): 
            influences += MaxInfluence.simulateAllCascades(graph, [], p=0.2)
        influences /= numRuns  
        
        influences2 = numpy.zeros(graph.vcount())
        for i in range(graph.vcount()): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([i]), numRuns, p=0.2)

             
        nptst.assert_array_almost_equal(influences, influences2, 2)
        
        #Test with some initial vertices 
        influences = numpy.zeros(graph.vcount())
        for i in range(numRuns): 
            influences += MaxInfluence.simulateAllCascades(graph, [0], p=0.2)
        influences /= numRuns  
        
        influences2 = numpy.zeros(graph.vcount())
        for i in range(graph.vcount()): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([0, i]), numRuns, p=0.2)
        
        print(influences, influences2)
        nptst.assert_array_almost_equal(influences, influences2, 2)
        
        influences = numpy.zeros(graph.vcount())
        for i in range(numRuns): 
            influences += MaxInfluence.simulateAllCascades(graph, [0, 5], p=0.2)
        influences /= numRuns  
        
        influences2 = numpy.zeros(graph.vcount())
        for i in range(graph.vcount()): 
            influences2[i] = MaxInfluence.simulateCascades(graph, set([0, 5, i]), numRuns, p=0.2)
        
        print(influences, influences2)
        nptst.assert_array_almost_equal(influences, influences2, 2)

if __name__ == '__main__':
    unittest.main()

