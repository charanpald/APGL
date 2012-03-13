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
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()