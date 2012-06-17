
from apgl.graph.test.MatrixGraphTest import MatrixGraphTest
from apgl.graph.VertexList import VertexList
from apgl.util import *
import unittest
import apgl
import scipy.sparse as sparse 

try: 
    from apgl.graph.PySparseGraph import PySparseGraph
except ImportError:
    pass 

@apgl.skipIf(not apgl.checkImport('pysparse'), 'No module pysparse')
class PySparseGraphTest(unittest.TestCase, MatrixGraphTest):
    

    def setUp(self):
        self.GraphType = PySparseGraph
        self.initialise()

    def testInit(self):
        numVertices = 0
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = PySparseGraph(vList)

        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = PySparseGraph(vList)

        self.assertRaises(ValueError, PySparseGraph, [])
        
    def testSetWeightMatrixSparse(self): 
        numVertices = 10
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph[0, 1] = 1
        
        W = sparse.lil_matrix((numVertices, numVertices))
        
        W[2, 1] = 1 
        W[1, 2] = 1 
        W[3, 8] = 0.2 
        W[8, 3] = 0.2 
        
        self.assertEquals(graph[0, 1], 1)   
        
        graph.setWeightMatrixSparse(W)
        
        self.assertEquals(graph[0, 1], 0)
        self.assertEquals(graph[2, 1], 1)
        self.assertEquals(graph[3, 8], 0.2)
        self.assertEquals(graph.getNumEdges(), 2)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()