import numpy 
import unittest
import logging
import ctypes
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 

from exp.influence2.GraphRanker import GraphRanker 

class  GraphRankerTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
        numpy.set_printoptions(suppress=True, precision=3)
        
        
    def testRankedList(self): 
        graph = igraph.Graph(directed=True)        
        graph.add_vertices(5)
        
        graph.add_edges([(0,1), (0,2), (1, 3), (2,3), (2,4)])   
        graph.es["invWeight"] = [1, 1, 1, 1, 1]
        graph.es["weight"] = [1, 1, 1, 1, 1]
        
        ranker = GraphRanker()
        outputList = ranker.vertexRankings(graph, [0, 1, 2])
        
        
        for lst in outputList:
            self.assertEquals(set(lst), set([0, 1,2])) 
            self.assertEquals(len(lst), 3)
            
    def testRestrictRankedList(self): 
        ranker = GraphRanker()
        
        list1 = [4, 2, 1,7,3, 8, 9, 12]
        list2 = [8, 9, 2]
        
        outputList = ranker.restrictRankedList(list1, list2)
        self.assertEquals(outputList, [2, 8, 9])
        
        outputList = ranker.restrictRankedList(list1, [3, 1, 12])
        self.assertEquals(outputList, [1, 3, 12])
    
if __name__ == '__main__':
    unittest.main()
