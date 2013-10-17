from apgl.graph.DictGraph import DictGraph
from apgl.util.Util import Util 
import unittest
import numpy 
import logging
import numpy.testing as nptst

class DictGraphTest(unittest.TestCase):
    def setUp(self):
        self.graph = DictGraph()
        self.graph.addEdge(0, 1, 1)
        self.graph.addEdge(1, 3, 1)
        self.graph.addEdge(0, 2, 2)
        self.graph.addEdge(2, 3, 5)
        self.graph.addEdge(0, 4, 1)
        self.graph.addEdge(3, 4, 1)
        self.graph.setVertex(5, None)
        
        self.graph2 = DictGraph(False)
        self.graph2.addEdge(0, 1, 1)
        self.graph2.addEdge(1, 3, 1)
        self.graph2.addEdge(0, 2, 2)
        self.graph2.addEdge(2, 3, 5)
        self.graph2.addEdge(0, 4, 1)
        self.graph2.addEdge(3, 4, 1)
        self.graph2.setVertex(5, 1)

    def testInit(self):
        dictGraph = DictGraph()

    def testAddEdge(self):
        dictGraph = DictGraph()
        dictGraph.addEdge("A", "B", [1,2,3])
        dictGraph.addEdge("A", "C", "HelloThere")
        dictGraph.addEdge(12, 8, [1,2,3, 12])

        self.assertEquals(dictGraph.getEdge("A", "B"), [1,2,3])
        self.assertEquals(dictGraph.getEdge("B", "A"), [1,2,3])
        self.assertEquals(dictGraph.getEdge("A", "C"), "HelloThere")
        self.assertEquals(dictGraph.getEdge("C", "A"), "HelloThere")
        self.assertEquals(dictGraph.getEdge(12, 8), [1,2,3, 12])
        self.assertEquals(dictGraph.getEdge(8, 12), [1,2,3, 12])

        dictGraph.addEdge(2, 8)

        dictGraph = DictGraph(False)
        dictGraph.addEdge("A", "B", [1,2,3])
        dictGraph.addEdge("A", "C", "HelloThere")
        dictGraph.addEdge(12, 8, [1,2,3, 12])

        self.assertEquals(dictGraph.getEdge("A", "B"), [1,2,3])
        self.assertEquals(dictGraph.getEdge("B", "A"), None)
        self.assertEquals(dictGraph.getEdge("A", "C"), "HelloThere")
        self.assertEquals(dictGraph.getEdge("C", "A"), None)
        self.assertEquals(dictGraph.getEdge(12, 8), [1,2,3, 12])
        self.assertEquals(dictGraph.getEdge(8, 12), None)

        dictGraph.addEdge(2, 8)

        #Test directed graphs 

    def testRemoveEdge(self):
        dictGraph = DictGraph()
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.addEdge(3, 4, 1)

        self.assertEquals(dictGraph.getEdge(1, 2), 12)
        self.assertEquals(dictGraph.getEdge(1, 3), 18)
        self.assertEquals(dictGraph.getEdge(3, 4), 1)

        dictGraph.removeEdge(1, 3)

        self.assertEquals(dictGraph.getEdge(1, 3), None)
        self.assertEquals(dictGraph.getEdge(1, 2), 12)
        self.assertEquals(dictGraph.getEdge(3, 4), 1)

        #Some tests on directed graphs
        dictGraph = DictGraph(False)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(2, 1, 12)

        dictGraph.removeEdge(1, 2)
        self.assertEquals(dictGraph.getEdge(1, 2), None)
        self.assertEquals(dictGraph.getEdge(2, 1), 12)

    def testIsUndirected(self):
        dictGraph = DictGraph(True)
        self.assertEquals(dictGraph.isUndirected(), True)

        dictGraph = DictGraph(False)
        self.assertEquals(dictGraph.isUndirected(), False)

    def testGetNumEdges(self):
        dictGraph = DictGraph(True)
        self.assertEquals(dictGraph.getNumEdges(), 0)

        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.addEdge(3, 4, 1)
        self.assertEquals(dictGraph.getNumEdges(), 3)

        dictGraph.addEdge(3, 4, 1)
        self.assertEquals(dictGraph.getNumEdges(), 3)

        dictGraph.addEdge(3, 5, 1)
        self.assertEquals(dictGraph.getNumEdges(), 4)

        dictGraph.addEdge(3, 3, 1)
        self.assertEquals(dictGraph.getNumEdges(), 5)

        #Identical tests with directed graphs
        dictGraph = DictGraph(False)
        self.assertEquals(dictGraph.getNumEdges(), 0)

        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.addEdge(3, 4, 1)
        self.assertEquals(dictGraph.getNumEdges(), 3)

        dictGraph.addEdge(3, 4, 1)
        self.assertEquals(dictGraph.getNumEdges(), 3)

        dictGraph.addEdge(3, 5, 1)
        self.assertEquals(dictGraph.getNumEdges(), 4)

        dictGraph.addEdge(3, 3, 1)
        self.assertEquals(dictGraph.getNumEdges(), 5)

    def testGetEdge(self):
        dictGraph = DictGraph(True)
        dictGraph.addEdge(1, 2, 12)

        self.assertEquals(dictGraph.getEdge(1, 2), 12)
        self.assertEquals(dictGraph.getEdge(2, 1), 12)
        self.assertEquals(dictGraph.getEdge(2, 2), None)
        self.assertRaises(ValueError, dictGraph.getEdge, 5, 8)

        dictGraph = DictGraph(False)
        dictGraph.addEdge(1, 2, 12)

        self.assertEquals(dictGraph.getEdge(1, 2), 12)
        self.assertEquals(dictGraph.getEdge(2, 1), None)

    def testGetNeighbours(self):
        dictGraph = DictGraph(True)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.addEdge(1, 4, 1)
        dictGraph.addEdge(3, 4, 1)
        dictGraph.addEdge(2, 2, 1)
        dictGraph.setVertex(5, 12)

        self.assertEquals(dictGraph.neighbours(1), [2, 3, 4])
        self.assertEquals(dictGraph.neighbours(3), [1, 4])
        self.assertEquals(dictGraph.neighbours(2), [1, 2])
        self.assertEquals(dictGraph.neighbours(5), [])

        #Directed graphs
        dictGraph = DictGraph(False)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.addEdge(1, 4, 1)
        dictGraph.addEdge(3, 4, 1)
        dictGraph.addEdge(2, 2, 1)
        dictGraph.setVertex(5, 12)

        self.assertEquals(dictGraph.neighbours(1), [2,3,4])
        self.assertEquals(dictGraph.neighbours(3), [4])
        self.assertEquals(dictGraph.neighbours(2), [2])
        self.assertEquals(dictGraph.neighbours(5), [])

    def testGetVertex(self):
        dictGraph = DictGraph(True)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.setVertex(5, 12)

        self.assertEquals(dictGraph.getVertex(1), None)
        self.assertEquals(dictGraph.getVertex(2), None)
        self.assertEquals(dictGraph.getVertex(3), None)
        self.assertEquals(dictGraph.getVertex(5), 12)

        self.assertRaises(ValueError, dictGraph.getVertex, 4)

        #Directed graphs
        dictGraph = DictGraph(False)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.setVertex(5, 12)

        self.assertEquals(dictGraph.getVertex(1), None)
        self.assertEquals(dictGraph.getVertex(2), None)
        self.assertEquals(dictGraph.getVertex(3), None)
        self.assertEquals(dictGraph.getVertex(5), 12)

        self.assertRaises(ValueError, dictGraph.getVertex, 4)

    def testAddVertex(self):
        dictGraph = DictGraph(True)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.setVertex(5, 12)

        self.assertEquals(dictGraph.getVertex(5), 12)

        dictGraph.setVertex(5, 22)
        self.assertEquals(dictGraph.getVertex(5), 22)

        dictGraph.addEdge(5, 11, 18)
        self.assertEquals(dictGraph.getVertex(5), 22)

    def testGetAllVertexIds(self):
        dictGraph = DictGraph(True)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)
        dictGraph.setVertex(5, 12)

        self.assertEquals(dictGraph.getAllVertexIds(), [1, 2, 3, 5])

    def testGetAllEdges(self):
        dictGraph = DictGraph(True)
        dictGraph.setVertex(5, 12)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(1, 3, 18)

        edges = dictGraph.getAllEdges()

        self.assertEquals(len(edges), 2)
        self.assertTrue((1,2) in edges)
        self.assertTrue((1,3) in edges)

        dictGraph = DictGraph(False)
        dictGraph.setVertex(5, 12)
        dictGraph.addEdge(1, 2, 12)
        dictGraph.addEdge(2, 1, 12)
        dictGraph.addEdge(1, 3, 18)

        edges = dictGraph.getAllEdges()

        self.assertEquals(len(edges), 3)
        self.assertTrue((1,2) in edges)
        self.assertTrue((2,1) in edges)
        self.assertTrue((1,3) in edges)

    def testDensity(self):
        numVertices = 10 
        graph = DictGraph(True)
        for i in range(numVertices):
            graph.setVertex(i, 0)

        graph.addEdge(0, 1)
        self.assertEquals(graph.density(), float(1)/45)

        graph.addEdge(0, 2)
        self.assertEquals(graph.density(), float(2)/45)

        graph = DictGraph(False)
        for i in range(numVertices):
            graph.setVertex(i, 0)
        graph.addEdge(0, 1)
        self.assertEquals(graph.density(), float(1)/90)

        graph.addEdge(0, 2)
        self.assertEquals(graph.density(), float(2)/90)

        #Test a graph with 1 vertex
        graph = DictGraph(True)
        graph.setVertex(0, 12)

        self.assertEquals(graph.density(), 0)

        graph.addEdge(0, 0)
        self.assertEquals(graph.density(), 1)

    def testSetVertices(self):
        graph = DictGraph()

        vertexIndices = [1, 2, 3]
        vertices = ["a", "b", "c"]

        graph.setVertices(vertexIndices, vertices)

        vertexIndices2 = graph.getAllVertexIds()
        vertices2 = graph.getVertices(vertexIndices2)

        self.assertEquals(vertexIndices, vertexIndices2)
        self.assertEquals(vertices, vertices2)

    def testGetWeightMatrix(self):
        graph = DictGraph()
        graph.addEdge("a", "b")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        graph.addEdge("d", "e")

        W = graph.getWeightMatrix()
        keys = graph.getAllVertexIds()

        for i in range(len(keys)):
            for j in range(len(keys)):
                if W[i, j] == 1:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), 1)
                else:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), None)

        #Try a directed graph
        graph = DictGraph(False)
        graph.addEdge("a", "b")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        graph.addEdge("d", "e")

        W = graph.getWeightMatrix()

        for i in range(len(keys)):
            for j in range(len(keys)):
                if W[i, j] == 1:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), 1)
                else:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), None)

    def testGetSparseWeightMatrix(self):
        graph = DictGraph()
        graph.addEdge("a", "b")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        graph.addEdge("d", "e")

        W = graph.getSparseWeightMatrix()
        keys = graph.getAllVertexIds()

        for i in range(len(keys)):
            for j in range(len(keys)):
                if W[i, j] == 1:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), 1)
                else:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), None)

        #Try a directed graph
        graph = DictGraph(False)
        graph.addEdge("a", "b")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        graph.addEdge("d", "e")

        W = graph.getSparseWeightMatrix()

        for i in range(len(keys)):
            for j in range(len(keys)):
                if W[i, j] == 1:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), 1)
                else:
                    self.assertEquals(graph.getEdge(keys[i], keys[j]), None)

    def testGetAllEdgeIndices(self):
        graph = DictGraph()
        graph.addEdge("a", "b")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        graph.addEdge("d", "e")

        edgeIndices = graph.getAllEdgeIndices() 
        keys = graph.getAllVertexIds() 

        self.assertEquals(edgeIndices.shape[0], graph.getNumEdges())
        for i in range(edgeIndices.shape[0]):
            self.assertTrue(graph.getEdge(keys[int(edgeIndices[i, 0])], keys[edgeIndices[i, 1]]) == 1)

        graph = DictGraph(False)
        graph.addEdge("a", "b")
        graph.addEdge("b", "a")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        graph.addEdge("d", "e")

        edgeIndices = graph.getAllEdgeIndices() 
        keys = graph.getAllVertexIds()
        self.assertEquals(edgeIndices.shape[0], graph.getNumEdges())
        for i in range(edgeIndices.shape[0]):
            self.assertTrue(graph.getEdge(keys[int(edgeIndices[i, 0])], keys[edgeIndices[i, 1]]) == 1)

    def testGetItem(self):
        graph = DictGraph()
        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)
        graph.addEdge(2, 4, 1)
        graph.addEdge(2, 3, 2)
        graph.setVertex(0, "abc")

        self.assertEquals(graph[1,1], 0.1)
        self.assertEquals(graph[1,3], 0.5)


    def testSetItem(self):
        graph = DictGraph()
        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)

        self.assertEquals(graph[1,3], 0.5)
        graph[1, 3] = 2
        self.assertEquals(graph[1,3], 2)

    def testAddEdges(self):
        graph = DictGraph()

        edgeList = [(1, 2), (2, 1), (5, 2), (8, 8)]

        graph.addEdges(edgeList)
        self.assertEquals(graph.getNumEdges(), 3)
        self.assertEquals(graph.getEdge(1, 2), 1)
        self.assertEquals(graph.getEdge(5, 2), 1)
        self.assertEquals(graph.getEdge(2, 1), 1)
        self.assertEquals(graph.getEdge(8, 8), 1)

        edgeValues = [1, 2, 3, 4]
        graph.addEdges(edgeList, edgeValues)
        self.assertEquals(graph.getEdge(1, 2), 2)
        self.assertEquals(graph.getEdge(5, 2), 3)
        self.assertEquals(graph.getEdge(2, 1), 2)
        self.assertEquals(graph.getEdge(8, 8), 4)

        #Now test directed graphs
        graph = DictGraph(False)
        graph.addEdges(edgeList)
        self.assertEquals(graph.getNumEdges(), 4)
        self.assertEquals(graph.getEdge(1, 2), 1)
        self.assertEquals(graph.getEdge(5, 2), 1)
        self.assertEquals(graph.getEdge(2, 1), 1)
        self.assertEquals(graph.getEdge(8, 8), 1)

        edgeValues = [1, 2, 3, 4]
        graph.addEdges(edgeList, edgeValues)
        self.assertEquals(graph.getEdge(1, 2), 1)
        self.assertEquals(graph.getEdge(5, 2), 3)
        self.assertEquals(graph.getEdge(2, 1), 2)
        self.assertEquals(graph.getEdge(8, 8), 4)


    def testSubgraph(self):
        graph = DictGraph()

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)
        graph.setVertex(0, "abc")
        graph.setVertex(3, "cde")

        self.assertEquals(graph.getNumEdges(), 5)

        subgraph = graph.subgraph([0, 1, 2])
        self.assertEquals(subgraph.getNumVertices(), 3)
        self.assertEquals(subgraph.getNumEdges(), 3)
        self.assertEquals(subgraph.isUndirected(), True)
        self.assertEquals(subgraph.getEdge(0, 1), 1)
        self.assertEquals(subgraph.getEdge(0, 2), 1)
        self.assertEquals(subgraph.getEdge(1, 2), 1)
        self.assertEquals(subgraph.getVertex(0), "abc")

        #Check the original graph is fine
        self.assertEquals(graph.getNumVertices(), 4)
        self.assertEquals(graph.getNumEdges(), 5)
        self.assertEquals(graph.getVertex(0), "abc")
        self.assertEquals(graph.getVertex(3), "cde")

        #Now a quick test for directed graphs
        graph = DictGraph(False)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        subgraph = graph.subgraph([0, 1, 2])
        self.assertEquals(subgraph.getNumEdges(), 3)
        self.assertEquals(subgraph.isUndirected(), False)
        self.assertEquals(subgraph.getEdge(0, 1), 1)
        self.assertEquals(subgraph.getEdge(0, 2), 1)
        self.assertEquals(subgraph.getEdge(1, 2), 1)

    def testNeighbourOf(self):
        graph = DictGraph(True)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        for i in range(4):
            self.assertEquals(graph.neighbours(i), graph.neighbourOf(i))

        #Now test directed graph
        graph = DictGraph(False)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        self.assertEquals(graph.neighbourOf(0), [])
        self.assertEquals(graph.neighbourOf(1), [0])
        self.assertEquals(graph.neighbourOf(2), [0,1])
        self.assertEquals(graph.neighbourOf(3), [0, 2])


    def testOutDegreeSequence(self):
        graph = DictGraph(True)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        degSeq, vertices = graph.outDegreeSequence()

        self.assertTrue((degSeq == numpy.array([ 3,  2,  3,  2.])).all())
        self.assertTrue(vertices == [0, 1, 2, 3])

        #Test results on a directed graph
        graph = DictGraph(False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        degSeq, vertices = graph.outDegreeSequence()
        self.assertTrue((degSeq == numpy.array([ 3,  1,  1,  0])).all())

    def testInDegreeSequence(self):
        graph = DictGraph(True)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        degSeq, vertices = graph.inDegreeSequence()

        self.assertTrue((degSeq == numpy.array([ 3,  2,  3,  2.])).all())
        self.assertTrue(vertices == [0, 1, 2, 3])

        #Test results on a directed graph
        graph = DictGraph(False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        degSeq, vertices = graph.inDegreeSequence()
        self.assertTrue((degSeq == numpy.array([ 0,  1,  2,  2])).all())

    def testVertexExists(self):
        graph = DictGraph(False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)

        self.assertTrue(graph.vertexExists(0))
        self.assertTrue(graph.vertexExists(1))
        self.assertTrue(graph.vertexExists(2))
        self.assertTrue(graph.vertexExists(3))
        self.assertFalse(graph.vertexExists(4))

    def testRemoveVertex(self):
        graph = DictGraph()
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)
        graph.addEdge(3, 4)

        graph.removeVertex(4)
        self.assertFalse(graph.vertexExists(4))
        self.assertFalse(graph.edgeExists(3, 4))
        
        graph.removeVertex(3)
        self.assertFalse(graph.vertexExists(3))
        self.assertFalse(graph.edgeExists(2, 3))
        self.assertFalse(graph.edgeExists(0, 3))
            
        graph.removeVertex(2)
        self.assertFalse(graph.vertexExists(2))
        self.assertFalse(graph.edgeExists(1, 2))
        self.assertFalse(graph.edgeExists(0, 2))
        
        self.assertTrue(graph.getAllVertexIds() == [0, 1])
        self.assertTrue(graph.getAllEdges() == [(0, 1)])
        
        #Try directed graph 
        graph = DictGraph(False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 0)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)
        graph.addEdge(3, 4)
        
        graph.removeVertex(0)

        self.assertFalse(graph.vertexExists(0))
        self.assertFalse(graph.edgeExists(0, 1))
        self.assertFalse(graph.edgeExists(0, 3))
        self.assertFalse(graph.edgeExists(1, 0))
        
        graph.removeVertex(2)
        self.assertFalse(graph.vertexExists(2))
        self.assertFalse(graph.edgeExists(1, 2))
        self.assertFalse(graph.edgeExists(2, 3))

        self.assertTrue(graph.getAllVertexIds() == [1, 3, 4])
        self.assertTrue(graph.getAllEdges() == [(3, 4)])
        
    def testToSparseGraph(self): 
        graph = DictGraph()
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(2, 3)
        graph.addEdge(3, 4)
        
        graph2 = graph.toSparseGraph()
        
        self.assertEquals(graph2[0, 1], 1)
        self.assertEquals(graph2[0, 2], 1)
        self.assertEquals(graph2[0, 3], 1)
        self.assertEquals(graph2[2, 1], 1)
        self.assertEquals(graph2[2, 3], 1)
        self.assertEquals(graph2[3, 4], 1)

    def testDepthFirstSearch(self):
        graph = DictGraph()
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)
        graph.addEdge(2, 6)
        graph.addEdge(4, 5)

        self.assertEquals(graph.depthFirstSearch(0), [0,1,2,6,3])
        self.assertEquals(graph.depthFirstSearch(1), [1,0,2,6,3])
        self.assertEquals(graph.depthFirstSearch(6), [6,2,1,0,3])
        self.assertEquals(graph.depthFirstSearch(4), [4, 5])
        self.assertEquals(graph.depthFirstSearch(5), [5, 4])

    def testBreadthFirstSearch(self):
        graph = DictGraph()
        graph.addEdge(0, 1)
        graph.addEdge(0, 7)
        graph.addEdge(7, 8)
        graph.addEdge(7, 9)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)
        graph.addEdge(2, 6)
        graph.addEdge(4, 5)

        self.assertEquals(graph.breadthFirstSearch(0), [0,1, 7,2,3,8,9,6])
        self.assertEquals(graph.breadthFirstSearch(1), [1,0,2,3,7,6,8,9])
        self.assertEquals(graph.breadthFirstSearch(6), [6, 2,1,0,3,7,8,9])
        self.assertEquals(graph.breadthFirstSearch(4), [4, 5])
        self.assertEquals(graph.breadthFirstSearch(5), [5, 4])
        self.assertEquals(graph.breadthFirstSearch(7), [7, 0, 8, 9, 1, 2, 3, 6])    

    def testDegreeSequence(self): 
        graph = DictGraph() 
        graph.setVertex("a", 10)
        graph["b", "c"] = 1
        graph["b", "d"] = 1
        graph["d", "e"] = 1
        graph["e", "e"] = 1
                
        degreeDict = {}
        degreeDict2 = {"a": 0, "b": 2, "c": 1, "d": 2, "e": 3}
        
        for i, id in enumerate(graph.getAllVertexIds()): 
            degreeDict[id] = graph.degreeSequence()[i]
            
        self.assertEquals(degreeDict, degreeDict2)

    def testGetNumDirEdges(self):
        graph = DictGraph()
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.1)

        self.assertTrue(graph.getNumDirEdges() == 4)
        graph.addEdge(1, 1)
        self.assertTrue(graph.getNumDirEdges() == 5)

        graph = DictGraph(False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)

        self.assertTrue(graph.getNumDirEdges() == 2)
        graph.addEdge(1, 1)
        self.assertTrue(graph.getNumDirEdges() == 3)

    def testDijkstrasAlgorithm(self):
        graph = DictGraph()

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(1, 3, 1)
        graph.addEdge(2, 4, 1)
        graph.setVertex(4, 1)

        self.assertTrue((graph.dijkstrasAlgorithm(0) == numpy.array([0, 1, 2, 2, 3])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(1) == numpy.array([1, 0, 1, 1, 2])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(2) == numpy.array([2, 1, 0, 2, 1])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(3) == numpy.array([2, 1, 2, 0, 3])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(4) == numpy.array([3, 2, 1, 3, 0])).all())

        
        #Test a graph which has an isolated node
        graph = DictGraph()
        graph.setVertex(5, 1)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(1, 3, 1)

        self.assertTrue((graph.dijkstrasAlgorithm(0) == numpy.array([0, 1, 2, 2, numpy.inf])).all())

        #Test a graph in a ring
        graph = DictGraph()

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)
        graph.addEdge(3, 4, 1)
        graph.addEdge(4, 0, 1)

        self.assertTrue((graph.dijkstrasAlgorithm(0) == numpy.array([0, 1, 2, 2, 1])).all())
        
        #Try case in which vertex ids are not numbers 
        graph = DictGraph()

        graph.addEdge("a", "b", 1)
        graph.addEdge("b", "c", 1)
        graph.addEdge("b", "d", 1)
        graph.addEdge("c", "e", 1)

        inds = Util.argsort(graph.getAllVertexIds())
        self.assertTrue((graph.dijkstrasAlgorithm("a")[inds] == numpy.array([0, 1, 2, 2, 3])).all())
        self.assertTrue((graph.dijkstrasAlgorithm("b")[inds] == numpy.array([1, 0, 1, 1, 2])).all())
        self.assertTrue((graph.dijkstrasAlgorithm("c")[inds] == numpy.array([2, 1, 0, 2, 1])).all())
        self.assertTrue((graph.dijkstrasAlgorithm("d")[inds] == numpy.array([2, 1, 2, 0, 3])).all())
        self.assertTrue((graph.dijkstrasAlgorithm("e")[inds] == numpy.array([3, 2, 1, 3, 0])).all())

    def testAdjacencyList(self): 
        graph = DictGraph()
        graph.addEdge("a", "b", 1)
        graph.addEdge("b", "c", 1)
        graph.addEdge("b", "d", 1)
        graph.addEdge("c", "e", 1)
        graph.setVertex("f", 1)
 
        neighbourIndices, neighbourWeights = graph.adjacencyList()   
 
        vertexIds = graph.getAllVertexIds()

        for i in range(len(neighbourIndices)): 
            for k, j in enumerate(neighbourIndices[i]): 
                self.assertTrue(graph.edgeExists(vertexIds[i], vertexIds[j]))  
                self.assertEquals(graph[vertexIds[i], vertexIds[j]], neighbourWeights[i][k])
         
    def testFindAllDistances(self):
        P = self.graph.findAllDistances()

        P2 = numpy.zeros((self.graph.size, self.graph.size))
        P2[0, :] = numpy.array([0, 1, 2, 2, 1, numpy.inf])
        P2[1, :] = numpy.array([1, 0, 3, 1, 2, numpy.inf])
        P2[2, :] = numpy.array([2, 3, 0, 4, 3, numpy.inf])
        P2[3, :] = numpy.array([2, 1, 4, 0, 1, numpy.inf])
        P2[4, :] = numpy.array([1, 2, 3, 1, 0, numpy.inf])
        P2[5, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0])

        self.assertTrue((P == P2).all())

        #Now test the directed graph
        P = self.graph2.findAllDistances()

        P2 = numpy.zeros((self.graph.size, self.graph.size))
        P2[0, :] = numpy.array([0, 1, 2, 2, 1, numpy.inf])
        P2[1, :] = numpy.array([numpy.inf, 0, numpy.inf, 1, 2, numpy.inf])
        P2[2, :] = numpy.array([numpy.inf, numpy.inf, 0, 5, 6, numpy.inf])
        P2[3, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, 0, 1, numpy.inf])
        P2[4, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0, numpy.inf])
        P2[5, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0])

        self.assertTrue((P == P2).all())

    def testToIGraph(self): 
        try:
            import igraph
        except ImportError as error:
            logging.debug(error)
            return        
        
        graph = DictGraph()
        
        graph["a", "b"] = 1
        graph["b", "c"] = 2
        
        ig = graph.toIGraph()
        
        self.assertEquals(len(ig.vs), 3) 
        self.assertEquals(ig[0, 2], 1) 
        self.assertEquals(ig[1, 2], 1)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    