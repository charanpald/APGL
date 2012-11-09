from apgl.graph.DictGraph import DictGraph
import unittest
import numpy 

class DictGraphTest(unittest.TestCase):
    def setUp(self):
        pass

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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    