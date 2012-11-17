
from apgl.graph.test.MatrixGraphTest import MatrixGraphTest
from apgl.graph.VertexList import VertexList
from apgl.util import *
import unittest
import apgl
import scipy.sparse 
import numpy 


try: 
    from apgl.graph.PySparseGraph import PySparseGraph
    from pysparse import spmatrix
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
        self.assertRaises(ValueError, PySparseGraph, vList, 1)
        self.assertRaises(ValueError, PySparseGraph, vList, True, 1)

        #Now test invalid values of W
        W = scipy.sparse.csr_matrix((numVertices, numVertices))
        self.assertRaises(ValueError, PySparseGraph, vList, True, W)

        W = numpy.zeros((numVertices+1, numVertices))
        self.assertRaises(ValueError, PySparseGraph, vList, True, W)

        W = numpy.zeros((numVertices, numVertices))
        W[0, 1] = 1
        self.assertRaises(ValueError, PySparseGraph, vList, True, W)

        W = spmatrix.ll_mat(numVertices, numVertices)
        graph = PySparseGraph(vList, W=W)

        self.assertTrue(isinstance(W, spmatrix.LLMatType))
        
        #Test intialising with non-empty graph 
        numVertices = 10 
        W = spmatrix.ll_mat(numVertices, numVertices)
        W[1, 0] = 1.1 
        W[0, 1] = 1.1 
        graph = PySparseGraph(numVertices, W=W)
        
        self.assertEquals(graph[1, 0], 1.1)
        
        #Test just specifying number of vertices 
        graph = PySparseGraph(numVertices)
        self.assertEquals(graph.size, numVertices)
        
        
    def testSetWeightMatrixSparse(self): 
        numVertices = 10
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph[0, 1] = 1
        
        W = scipy.sparse.lil_matrix((numVertices, numVertices))
        
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