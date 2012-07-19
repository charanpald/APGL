
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.graph.VertexList import VertexList
from apgl.graph.test.MatrixGraphTest import MatrixGraphTest
from apgl.util.Util import Util
import scipy.sparse as sparse
import unittest
import numpy
import scipy
import logging

class SparseGraphTest(unittest.TestCase, MatrixGraphTest):
    def setUp(self):
        self.GraphType = SparseGraph
        self.initialise()

    def testInit(self):
        numVertices = 0
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)
        self.assertEquals(graph.weightMatrixType(), scipy.sparse.csr_matrix)

        self.assertRaises(ValueError, SparseGraph, [])
        self.assertRaises(ValueError, SparseGraph, vList, 1)
        self.assertRaises(ValueError, SparseGraph, vList, True, 1)

        #Now test invalid values of W
        W = numpy.zeros((numVertices, numVertices))
        self.assertRaises(ValueError, SparseGraph, vList, True, W)

        W = scipy.sparse.lil_matrix((numVertices+1, numVertices))
        self.assertRaises(ValueError, SparseGraph, vList, True, W)

        W = scipy.sparse.lil_matrix((numVertices, numVertices))
        W[0, 1] = 1
        self.assertRaises(ValueError, SparseGraph, vList, True, W)

        W = scipy.sparse.lil_matrix((numVertices, numVertices))
        graph = SparseGraph(vList, W=W)

        self.assertEquals(graph.weightMatrixType(), scipy.sparse.lil_matrix)

    def testNativeAdjacencyMatrix(self):
        numVertices = 10 
        graph = SparseGraph(GeneralVertexList(numVertices))

        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)
        graph.addEdge(2, 5, 1)
        graph.addEdge(7, 0, 2)

        A = graph.nativeAdjacencyMatrix()
        self.assertEquals(A[0, 7], 1)
        self.assertEquals(A[7, 0], 1)
        self.assertEquals(A[1, 3], 1)
        self.assertEquals(A[3, 1], 1)
        self.assertEquals(A[1, 1], 1)
        self.assertEquals(A[2, 5], 1)
        self.assertEquals(A[5, 2], 1)
        self.assertEquals(A.getnnz(), 7)

        graph = SparseGraph(GeneralVertexList(numVertices), False)
        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)
        graph.addEdge(2, 5, 1)

        A = graph.nativeAdjacencyMatrix()
        self.assertEquals(A[1, 3], 1)
        self.assertEquals(A[1, 1], 1)
        self.assertEquals(A[2, 5], 1)
        self.assertEquals(A.getnnz(), 3)

    def testConcat(self):
        numVertices = 5
        graph = SparseGraph(GeneralVertexList(numVertices))
        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)
        graph.addEdge(2, 4, 1)
        graph.addEdge(2, 3, 2)
        graph.setVertex(0, "abc")

        graph2 = SparseGraph(GeneralVertexList(numVertices))
        graph2.addEdge(1, 1)
        graph2.addEdge(1, 4)
        graph2.setVertex(1, "def")

        graph3 = graph.concat(graph2)

        self.assertTrue(graph3.getNumVertices, 10)
        self.assertEquals(graph3.getVertex(0), "abc")
        self.assertEquals(graph3.getVertex(6), "def")
        self.assertEquals(graph3.getEdge(1, 1), 0.1)
        self.assertEquals(graph3.getEdge(1, 3), 0.5)
        self.assertEquals(graph3.getEdge(2, 4), 1)
        self.assertEquals(graph3.getEdge(2, 3), 2)

        self.assertEquals(graph3.getEdge(6, 6), 1)
        self.assertEquals(graph3.getEdge(6, 9), 1)

    def testNormalisedLaplacianSym2(self):
        numVertices = 10
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 9)
        graph.addEdge(1, 1)
        graph.addEdge(1, 5)

        L = graph.normalisedLaplacianSym(sparse=True)

        W = graph.getWeightMatrix()
        L2 = numpy.zeros((numVertices, numVertices))
        d = graph.outDegreeSequence()

        for i in range(numVertices):
            for j in range(numVertices):
                if d[i] != 0 and d[j]!= 0:
                    Wij = W[i, j]/(numpy.sqrt(d[i]*d[j]))
                else:
                    Wij = 0

                if i == j:
                    L2[i, j] = 1 - Wij
                else:
                    L2[i, j] = -Wij

                self.assertAlmostEquals(L[i, j], L2[i, j])
    
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
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()