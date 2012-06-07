
from apgl.graph.test.MatrixGraphTest import MatrixGraphTest
from apgl.graph.VertexList import VertexList
from apgl.util import *
import unittest
import apgl
import pickle

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
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()