'''
Created on 3 Feb 2010 

@author: charanpal
'''
from apgl.graph.SparseMultiGraph import SparseMultiGraph
from apgl.graph.VertexList import VertexList
import unittest
import numpy


class SparseMultiGraphTest(unittest.TestCase):
    def setUp(self):
        self.numVertices = 10
        self.numFeatures = 3
        self.maxEdgeTypes = 3
        self.vList = VertexList(self.numVertices, self.numFeatures)

        self.sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes)

    def testInit(self):
        sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes)
        sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes, False)

        self.assertRaises(ValueError, SparseMultiGraph, self.vList, 0)
        self.assertRaises(ValueError, SparseMultiGraph, self.vList, -1)

    def testAddEdge(self):
        self.sMultiGraph.addEdge(0, 1, 0, 5)
        self.sMultiGraph.addEdge(0, 1, 1, 2)
        self.sMultiGraph.addEdge(5, 3, 2, 12)

        self.assertEquals(self.sMultiGraph.getEdge(0, 1, 0), 5)
        self.assertEquals(self.sMultiGraph.getEdge(0, 1, 1), 2)
        self.assertEquals(self.sMultiGraph.getEdge(5, 3, 2), 12)
        self.assertEquals(self.sMultiGraph.getEdge(5, 3, 1), None)
        self.assertEquals(self.sMultiGraph.getEdge(8, 2, 1), None)

        self.assertRaises(ValueError, self.sMultiGraph.addEdge, 0, 1, -1)
        self.assertRaises(ValueError, self.sMultiGraph.addEdge, 0, 1, self.maxEdgeTypes)
        self.assertRaises(ValueError, self.sMultiGraph.addEdge, 0, self.numVertices, 0)
        self.assertRaises(ValueError, self.sMultiGraph.addEdge, self.numVertices, 0, 0)
        self.assertRaises(ValueError, self.sMultiGraph.addEdge, 0, -1, 0)
        self.assertRaises(ValueError, self.sMultiGraph.addEdge, -1, 0, 0)

    def testRemoveEdge(self):
        self.sMultiGraph.addEdge(2, 1, 0, 5)
        self.sMultiGraph.addEdge(2, 1, 1, 2)
        self.sMultiGraph.addEdge(6, 1, 1, 21)

        self.assertEquals(self.sMultiGraph.getEdge(2, 1, 0), 5)
        self.assertEquals(self.sMultiGraph.getEdge(2, 1, 1), 2)
        self.assertEquals(self.sMultiGraph.getEdge(6, 1, 1), 21)

        self.sMultiGraph.removeEdge(2, 1, 0)
        self.sMultiGraph.removeEdge(2, 1, 1)
        self.sMultiGraph.removeEdge(6, 1, 1)
        self.assertEquals(self.sMultiGraph.getEdge(2, 1, 0), None)
        self.assertEquals(self.sMultiGraph.getEdge(2, 1, 1), None)
        self.assertEquals(self.sMultiGraph.getEdge(6, 1, 1), None)

        #Remove an edge that does not exist 
        self.sMultiGraph.removeEdge(5, 5, 0)

        self.assertRaises(ValueError, self.sMultiGraph.removeEdge, 0, 1, -1)
        self.assertRaises(ValueError, self.sMultiGraph.removeEdge, 0, 1, self.maxEdgeTypes)
        self.assertRaises(ValueError, self.sMultiGraph.removeEdge, 0, self.numVertices, 0)
        self.assertRaises(ValueError, self.sMultiGraph.removeEdge, self.numVertices, 0, 0)
        self.assertRaises(ValueError, self.sMultiGraph.removeEdge, 0, -1, 0)
        self.assertRaises(ValueError, self.sMultiGraph.removeEdge, -1, 0, 0)

    def testGetNumEdges(self):
        sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes)

        self.assertEquals(sMultiGraph.getNumEdges(), 0)

        for i in range(self.maxEdgeTypes):
            self.assertEquals(sMultiGraph.getNumEdges(i), 0)

        sMultiGraph.addEdge(0, 1, 2)
        sMultiGraph.addEdge(0, 1, 1)

        self.assertEquals(sMultiGraph.getNumEdges(), 2)
        self.assertEquals(sMultiGraph.getNumEdges(1), 1)
        self.assertEquals(sMultiGraph.getNumEdges(2), 1)

        sMultiGraph.addEdge(0, 1, 0)
        sMultiGraph.addEdge(0, 2, 0)
        self.assertEquals(sMultiGraph.getNumEdges(), 4)
        self.assertEquals(sMultiGraph.getNumEdges(0), 2)
        
    def testGetNumVertices(self):
        self.assertEquals(self.sMultiGraph.getNumVertices(), self.numVertices)

    def testIsUndirected(self):
        self.assertEquals(self.sMultiGraph.isUndirected(), True)

        sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes, False)

        self.assertEquals(sMultiGraph.isUndirected(), False)
        
    def testGetNeighbours(self):
        sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes, True)

        sMultiGraph.addEdge(0, 1, 2)
        sMultiGraph.addEdge(0, 3, 1)
        sMultiGraph.addEdge(2, 1, 0)
        sMultiGraph.addEdge(1, 4, 0)
        sMultiGraph.addEdge(9, 4, 0)
        sMultiGraph.addEdge(9, 4, 1)

        self.assertEquals(set(sMultiGraph.neighbours(0)), set([1, 3]))
        self.assertEquals(set(sMultiGraph.neighbours(1)), set([0, 2, 4]))
        self.assertEquals(sMultiGraph.neighbours(3), [0])
        self.assertEquals(set(sMultiGraph.neighbours(4)), set([9, 1]))
        self.assertEquals(sMultiGraph.neighbours(9), [4])

    def testGetNeighboursByEdgeType(self):
        sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes, True)

        sMultiGraph.addEdge(0, 1, 2)
        sMultiGraph.addEdge(0, 3, 1)
        sMultiGraph.addEdge(2, 1, 0)
        sMultiGraph.addEdge(1, 4, 0)
        sMultiGraph.addEdge(9, 4, 0)
        sMultiGraph.addEdge(9, 4, 1)

        self.assertEquals(sMultiGraph.getNeighboursByEdgeType(0, 1), [3])
        self.assertEquals(sMultiGraph.getNeighboursByEdgeType(0, 2), [1])
        self.assertEquals(sMultiGraph.getNeighboursByEdgeType(0, 0), [])
        self.assertEquals(set(sMultiGraph.getNeighboursByEdgeType(4, 0)), set([1, 9]))


    def testGetAllVertexIds(self):
        self.assertTrue((self.sMultiGraph.getAllVertexIds() == numpy.array(list(range(0, self.numVertices)))).all())

    def testSetVertex(self):
        testVertex = numpy.array([3, 2, 9])
        self.sMultiGraph.setVertex(5, testVertex)

        self.assertTrue((self.sMultiGraph.getVertex(5) == testVertex).all())

    def testGetAllEdges(self):
        sMultiGraph = SparseMultiGraph(self.vList, self.maxEdgeTypes, False)

        self.assertEquals(sMultiGraph.getAllEdges().shape[0], 0)
        self.assertEquals(sMultiGraph.getAllEdges().shape[1], 3)

        sMultiGraph.addEdge(0, 1, 0)
        sMultiGraph.addEdge(0, 1, 1)
        sMultiGraph.addEdge(0, 2, 0)

        allEdges = numpy.array([[0,1,0], [0,2,0], [0,1,1]])

        self.assertTrue((sMultiGraph.getAllEdges() == allEdges).all())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

