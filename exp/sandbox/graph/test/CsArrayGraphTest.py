
from exp.sandbox.graph.CsArrayGraph import CsArrayGraph
from apgl.graph.VertexList import VertexList
from apgl.util import *
from apgl.graph.test.MatrixGraphTest import MatrixGraphTest
import unittest
import numpy


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
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()