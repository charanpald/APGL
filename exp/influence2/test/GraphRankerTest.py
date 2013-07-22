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
        
        outputList = GraphRanker.rankedLists(graph)
        
        
        for lst in outputList:
            self.assertEquals(len(lst), 5)
    
if __name__ == '__main__':
    unittest.main()
