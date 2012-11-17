'''
Created on 1 Jul 2009

@author: charanpal
'''
from apgl.graph.DenseGraph import DenseGraph
from apgl.graph.VertexList import VertexList
from apgl.util import *
from apgl.graph.test.MatrixGraphTest import MatrixGraphTest
import unittest
import numpy
import logging
import scipy.sparse 

class DenseGraphTest(unittest.TestCase, MatrixGraphTest):
    def setUp(self):
        self.GraphType = DenseGraph
        self.initialise()

    def testInit(self):
        numVertices = 0
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        graph = DenseGraph(vList)
        self.assertEquals(graph.weightMatrixDType(), numpy.float64)

        graph = DenseGraph(vList, dtype=numpy.int16)
        self.assertEquals(graph.weightMatrixDType(), numpy.int16)
        
        numVertices = 0
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = DenseGraph(vList, dtype=numpy.int16)

        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = DenseGraph(vList, dtype=numpy.int16)
        self.assertEquals(type(graph.W), numpy.ndarray)

        self.assertRaises(ValueError, DenseGraph, [])
        self.assertRaises(ValueError, DenseGraph, vList, 1)
        self.assertRaises(ValueError, DenseGraph, vList, True, 1)

        #Now test invalid values of W
        W = scipy.sparse.csr_matrix((numVertices, numVertices))
        self.assertRaises(ValueError, DenseGraph, vList, True, W)

        W = numpy.zeros((numVertices+1, numVertices))
        self.assertRaises(ValueError, DenseGraph, vList, True, W)

        W = numpy.zeros((numVertices, numVertices))
        W[0, 1] = 1
        self.assertRaises(ValueError, DenseGraph, vList, True, W)

        W = numpy.zeros((numVertices, numVertices))
        graph = DenseGraph(vList, W=W)

        self.assertEquals(type(graph.W), numpy.ndarray)
        
        #Test intialising with non-empty graph 
        numVertices = 10 
        W = numpy.zeros((numVertices, numVertices))
        W[1, 0] = 1.1 
        W[0, 1] = 1.1 
        graph = DenseGraph(numVertices, W=W)
        
        self.assertEquals(graph[1, 0], 1.1)
        
        #Test just specifying number of vertices 
        graph = DenseGraph(numVertices)
        self.assertEquals(graph.size, numVertices)
        
        #Try creating a sparse matrix of dtype int 
        graph = DenseGraph(numVertices, dtype=numpy.int)
        self.assertEquals(graph.W.dtype, numpy.int)
        graph[0, 0] = 1.2 
        
        self.assertEquals(graph[0, 0], 1)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()