import apgl
from apgl.graph.VertexList import VertexList
from apgl.util import *
from apgl.graph.test.MatrixGraphTest import MatrixGraphTest
import unittest
import numpy

try: 
    from apgl.graph.CsArrayGraph import CsArrayGraph
except ImportError:
    pass 

@apgl.skipIf(not apgl.checkImport('sppy'), 'No module sppy')
class CsArrayGraphTest(unittest.TestCase, MatrixGraphTest):
    def setUp(self):
        self.GraphType = CsArrayGraph
        self.initialise()

    def testInit(self):
        numVertices = 0
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        graph = CsArrayGraph(vList)
        self.assertEquals(graph.weightMatrixDType(), numpy.float)

        graph = CsArrayGraph(vList, dtype=numpy.int16)
        self.assertEquals(graph.weightMatrixDType(), numpy.int16)
        
        #Test just specifying number of vertices 
        graph = CsArrayGraph(numVertices)
        self.assertEquals(graph.size, numVertices)
        
    def testAddVertices(self): 
        graph = CsArrayGraph(10)
        graph[3, 4] = 1
        
        graph.addVertices(5)
        
        self.assertEquals(graph[3,4], 1)
        self.assertEquals(graph.getNumEdges(), 1)
        self.assertEquals(graph.size, 15)
        
        graph[10, 11] = 0.1
        graph.addVertices(5)
        
        self.assertEquals(graph[3,4], 1)
        self.assertEquals(graph[10,11], 0.1)
        self.assertEquals(graph.getNumEdges(), 2)
        self.assertEquals(graph.size, 20)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()