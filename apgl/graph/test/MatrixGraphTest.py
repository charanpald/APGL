from apgl.graph.VertexList import VertexList
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.generator.BarabasiAlbertGenerator import BarabasiAlbertGenerator
from apgl.util.PathDefaults import PathDefaults
import numpy
import os
import logging
import pickle
import numpy.testing as nptst 
"""
A class which encapsulates common tests for classes than inherit from AbtractMatrixGraph.
"""

class MatrixGraphTest():
    def initialise(self):
        numpy.set_printoptions(suppress = True)
        numpy.random.seed(21)

        self.numVertices = 6
        self.numFeatures = 1
        self.vList = VertexList(self.numVertices, self.numFeatures)

        self.graph = self.GraphType(self.vList)
        self.graph.addEdge(0, 1, 1)
        self.graph.addEdge(1, 3, 1)
        self.graph.addEdge(0, 2, 2)
        self.graph.addEdge(2, 3, 5)
        self.graph.addEdge(0, 4, 1)
        self.graph.addEdge(3, 4, 1)

        self.graph2 = self.GraphType(self.vList, False)
        self.graph2.addEdge(0, 1, 1)
        self.graph2.addEdge(1, 3, 1)
        self.graph2.addEdge(0, 2, 2)
        self.graph2.addEdge(2, 3, 5)
        self.graph2.addEdge(0, 4, 1)
        self.graph2.addEdge(3, 4, 1)

    def testAddEdge(self):
        self.graph.addEdge(1, 5, 2)

        self.assertEquals(self.graph.getEdge(1,5), 2)
        self.assertEquals(self.graph.getEdge(5,1), 2)

        self.assertEquals(self.graph.getEdge(2,5), None)

        self.assertRaises(ValueError, self.graph.addEdge, 1, 3, 0)

    def testAddEdges(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = self.GraphType(vList)

        edgeIndexArray = numpy.array([[1,2], [2,3]])

        graph.addEdges(edgeIndexArray)
        self.assertEquals(graph.getEdge(1, 2), 1)
        self.assertEquals(graph.getEdge(3, 2), 1)
        self.assertEquals(graph.getEdge(2, 3), 1)
        self.assertEquals(graph.getEdge(2, 1), 1)
        self.assertEquals(graph.getNumEdges(), 2)

        graph = self.GraphType(vList, False)
        graph.addEdges(edgeIndexArray)

        self.assertEquals(graph.getNumEdges(), 2)
        self.assertEquals(graph.getEdge(1, 2), 1)
        self.assertEquals(graph.getEdge(2, 3), 1)

        edgeValues = numpy.array([0.1, 0.2])
        graph.addEdges(edgeIndexArray, edgeValues)
        self.assertEquals(graph.getEdge(1, 2), 0.1)
        self.assertEquals(graph.getEdge(2, 3), 0.2)

        graph = self.GraphType(vList)
        graph.addEdges(edgeIndexArray, edgeValues)
        self.assertEquals(graph.getEdge(1, 2), 0.1)
        self.assertEquals(graph.getEdge(2, 3), 0.2)
        self.assertEquals(graph.getEdge(2, 1), 0.1)
        self.assertEquals(graph.getEdge(3, 2), 0.2)

        edgeValues = numpy.array([0.1, 0.0])
        self.assertRaises(ValueError, graph.addEdges, edgeIndexArray, edgeValues)

    def testRemoveEdge(self):
        self.graph.addEdge(1, 5, 2)

        self.assertEquals(self.graph.getEdge(1,5), 2)
        self.assertEquals(self.graph.getEdge(5,1), 2)

        self.graph.removeEdge(1,5)

        self.assertEquals(self.graph.getEdge(1,5), None)
        self.assertEquals(self.graph.getEdge(5,2), None)

    def testNeighbours(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)

        graph.addEdge(1, 5, 2)
        graph.addEdge(1, 3, 5)
        graph.addEdge(1, 9, 1)
        graph.addEdge(2, 3, 2)

        self.assertTrue((numpy.sort(graph.neighbours(1)) == numpy.array([3,5,9])).all())
        self.assertTrue((graph.neighbours(2) == numpy.array([3])).all())
        self.assertTrue((numpy.sort(graph.neighbours(3)) == numpy.array([1,2])).all())
        self.assertTrue((graph.neighbours(4) == numpy.array([])).all())

        #Test this function for directed graphs
        graph = self.GraphType(vList, False)

        graph.addEdge(1, 5, 2)
        graph.addEdge(1, 3, 5)
        graph.addEdge(9, 1, 1)
        graph.addEdge(2, 3, 2)

        self.assertTrue((numpy.sort(graph.neighbours(1)) == numpy.array([3,5])).all())
        self.assertTrue((graph.neighbours(2) == numpy.array([3])).all())
        self.assertTrue((graph.neighbours(3) == numpy.array([])).all())
        self.assertTrue((graph.neighbours(9) == numpy.array([1])).all())

    def testNeighbourOf(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)

        graph.addEdge(1, 5, 2)
        graph.addEdge(1, 3, 5)
        graph.addEdge(1, 9, 1)
        graph.addEdge(2, 3, 2)

        self.assertTrue((graph.neighbourOf(1) == numpy.array([3,5,9])).all())
        self.assertTrue((graph.neighbourOf(2) == numpy.array([3])).all())
        self.assertTrue((graph.neighbourOf(3) == numpy.array([1,2])).all())
        self.assertTrue((graph.neighbourOf(4) == numpy.array([])).all())

        #Test this function for directed graphs
        graph = self.GraphType(vList, False)

        graph.addEdge(1, 5, 2)
        graph.addEdge(1, 3, 5)
        graph.addEdge(9, 1, 1)
        graph.addEdge(2, 3, 2)

        self.assertTrue((graph.neighbourOf(1) == numpy.array([9])).all())
        self.assertTrue((graph.neighbourOf(2) == numpy.array([])).all())
        self.assertTrue((graph.neighbourOf(3) == numpy.array([1, 2])).all())
        self.assertTrue((graph.neighbourOf(9) == numpy.array([])).all())

    def testClusteringCoefficient(self):
        numVertices = 3
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        #1st graph - take 3 nodes in a line
        graph = self.GraphType(vList)
        graph.addEdge(0, 1, 2)
        graph.addEdge(1, 2, 5)

        self.assertEqual(graph.clusteringCoefficient(), 0)

        #Now, form a triangle
        graph.addEdge(0, 2, 5)
        self.assertEqual(graph.clusteringCoefficient(), 1)

        #2nd Graph - taken from Newman
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 1, 2)
        graph.addEdge(0, 2, 2)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(2, 4, 2)

        self.assertEqual(graph.clusteringCoefficient(), float(3)/8)

        #3rd graph - has no edges
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        self.assertEqual(graph.clusteringCoefficient(), 0.0)

    def testDegreeDistribution(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        self.assertTrue((graph.degreeDistribution() == numpy.array([])).all())

        graph.addEdge(0, 1, 2)
        graph.addEdge(0, 2, 2)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(2, 4, 2)

        self.assertTrue((graph.degreeDistribution() == numpy.array([0, 2, 2, 0, 1])).all())

        #Try empty graph
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        self.assertTrue((graph.degreeDistribution() == numpy.array([5])).all())

        #Try a star like graph
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 0, 2)
        graph.addEdge(0, 1, 2)
        graph.addEdge(0, 2, 2)
        graph.addEdge(0, 3, 2)
        graph.addEdge(0, 4, 2)

        self.assertTrue((graph.degreeDistribution() == numpy.array([0, 4, 0, 0, 0, 1])).all())

        #Test obtaining a subgraph and then the degree distribution
        subGraph = graph.subgraph([0,1,2,3])
        #logging.debug(subGraph.degreeDistribution())


    def testDijkstrasAlgorithm(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 5)
        graph.addEdge(1, 2, 2)
        graph.addEdge(1, 3, 2)
        graph.addEdge(2, 4, 2)

        self.assertTrue((graph.dijkstrasAlgorithm(0) == numpy.array([0, 1, 2, 2, 3])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(1) == numpy.array([1, 0, 1, 1, 2])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(2) == numpy.array([2, 1, 0, 2, 1])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(3) == numpy.array([2, 1, 2, 0, 3])).all())
        self.assertTrue((graph.dijkstrasAlgorithm(4) == numpy.array([3, 2, 1, 3, 0])).all())

        #Test case which found a bug
        self.assertTrue((self.graph.dijkstrasAlgorithm(2, self.graph.adjacencyList()) == numpy.array([2,3,0,4,3, float('inf')])).all())
        
        #Test a graph which has an isolated node
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 5)
        graph.addEdge(1, 2, 2)
        graph.addEdge(1, 3, 2)

        self.assertTrue((graph.dijkstrasAlgorithm(0) == numpy.array([0, 1, 2, 2, numpy.inf])).all())

        #Test a graph in a ring
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 5)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(3, 4, 2)
        graph.addEdge(4, 0, 2)

        self.assertTrue((graph.dijkstrasAlgorithm(0) == numpy.array([0, 1, 2, 2, 1])).all())

    def testGeodesicDistance(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 5)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(3, 4, 2)
        graph.addEdge(4, 0, 2)

        P = graph.floydWarshall()
        self.assertEquals(graph.geodesicDistance(), 37/15.0)
        self.assertEquals(graph.geodesicDistance(P), 37/15.0)

        #Test a string of vertices
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)
        graph.addEdge(3, 4, 1)

        P = graph.floydWarshall()
        self.assertEquals(graph.geodesicDistance(), 4.0/3)
        self.assertEquals(graph.geodesicDistance(P), 4.0/3)

        #Test case with isolated node
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)

        P = graph.floydWarshall()
        self.assertEquals(graph.geodesicDistance(), 2.0/3)
        self.assertEquals(graph.geodesicDistance(P), 2.0/3)

        #Test directed graph
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)

        P = graph.floydWarshall()
        self.assertEquals(graph.geodesicDistance(), 4.0/25)
        self.assertEquals(graph.geodesicDistance(P), 4.0/25)

    def testHopCount(self):
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)

        self.assertTrue((graph.hopCount() == numpy.array([10, 16, 22])).all())

        graph.addEdge(0, 4)
        self.assertTrue((graph.hopCount() == numpy.array([10, 18, 30])).all())

        graph.addEdge(4, 5)
        self.assertTrue((graph.hopCount() == numpy.array([10, 20, 34, 40])).all())

        #Test case where we pass in P matrix
        P = graph.floydWarshall()
        self.assertTrue((graph.hopCount(P) == numpy.array([10, 20, 34, 40])).all())

        #Test a directed graph
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2, 0.1)
        graph.addEdge(0, 3)

        self.assertTrue((graph.hopCount() == numpy.array([10, 13])).all())

        P = graph.floydWarshall(False)
        self.assertTrue((graph.hopCount(P) == numpy.array([10, 13])).all())

        #Test empty graph and zero graph
        graph = self.GraphType(vList, True)
        self.assertTrue((graph.hopCount() == numpy.array([numVertices])).all())

        vList = VertexList(0, 0)
        graph = self.GraphType(vList, True)
        self.assertTrue((graph.hopCount() == numpy.array([])).all())





    def testHarmonicGeodesicDistance(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)
        graph.addEdge(3, 4, 1)
        graph.addEdge(4, 0, 1)

        self.assertEquals(graph.harmonicGeodesicDistance(), 2.0)
        P = graph.floydWarshall(True)
        self.assertEquals(graph.harmonicGeodesicDistance(P), 2.0)

        #Test a string of vertices
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)
        graph.addEdge(3, 4, 1)

        self.assertAlmostEquals(graph.harmonicGeodesicDistance(), 180/77.0, places=5)
        P = graph.floydWarshall(True)
        self.assertAlmostEquals(graph.harmonicGeodesicDistance(P), 180/77.0, places=5)

        #Test case with isolated node
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)

        self.assertAlmostEquals(graph.harmonicGeodesicDistance(), 45/13.0, places=5)
        P = graph.floydWarshall(True)
        self.assertAlmostEquals(graph.harmonicGeodesicDistance(P), 45/13.0, places=5)

        #Totally empty graph
        graph = self.GraphType(vList)
        self.assertEquals(graph.harmonicGeodesicDistance(), float('inf'))

        #Test use of indices 
        graph = self.GraphType(vList)
        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)
        graph.addEdge(3, 4, 1)

        P = graph.floydWarshall(True)
        inds = [0, 4]
        self.assertEquals(graph.harmonicGeodesicDistance(vertexInds=inds), 12.0)

        #Test directed graph
        graph = self.GraphType(vList, False)

        graph.addEdge(0, 1, 1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(2, 3, 1)
        graph.addEdge(3, 4, 1)

        P = graph.floydWarshall(True)
        self.assertAlmostEquals(graph.harmonicGeodesicDistance(P), 300/77.0, places=5)

    def testGetAllEdges(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph.addEdge(0, 1, 5)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(2, 2, 2)

        edges = graph.getAllEdges()

        self.assertEquals(edges.shape[0], 4)
        self.assertTrue((edges[0, :]== numpy.array([1,0])).all())
        self.assertTrue((edges[1, :]== numpy.array([2,1])).all())
        self.assertTrue((edges[2, :]== numpy.array([2,2])).all())
        self.assertTrue((edges[3, :]== numpy.array([3,2])).all())


        #Test a directed graph
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1, 5)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(2, 2, 2)
        graph.addEdge(2, 1, 2)

        edges = graph.getAllEdges()

        self.assertEquals(edges.shape[0], 5)
        self.assertTrue((edges[0, :]== numpy.array([0,1])).all())
        self.assertTrue((edges[1, :]== numpy.array([1,2])).all())
        self.assertTrue((edges[2, :]== numpy.array([2,1])).all())
        self.assertTrue((edges[3, :]== numpy.array([2,2])).all())
        self.assertTrue((edges[4, :]== numpy.array([2,3])).all())

        #Test graph with no edges
        graph = self.GraphType(vList)
        edges = graph.getAllEdges()
        self.assertEquals(edges.shape, (0, 2))

    def testGetNumEdges(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)
        graph.addEdge(0, 1, 1)

        self.assertEquals(graph.getNumEdges(), 1)

        graph.addEdge(3, 4, 1)
        graph.addEdge(3, 4, 1)

        self.assertEquals(graph.getNumEdges(), 2)

        graph.addEdge(5, 5, 1)
        self.assertEquals(graph.getNumEdges(), 3)

        graph.addEdge(8, 8, 1)
        graph.addEdge(8, 8, 1)
        self.assertEquals(graph.getNumEdges(), 4)

        #Now test directed graphs
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1, 1)

        self.assertEquals(graph.getNumEdges(), 1)

        graph.addEdge(3, 4, 1)
        graph.addEdge(3, 4, 1)

        self.assertEquals(graph.getNumEdges(), 2)

        graph.addEdge(5, 5, 1)
        self.assertEquals(graph.getNumEdges(), 3)

        graph.addEdge(8, 8, 1)
        graph.addEdge(8, 8, 1)
        self.assertEquals(graph.getNumEdges(), 4)

    def testGetNumVertices(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        self.assertEquals(graph.getNumVertices(), numVertices)

    def testGetEdge(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(2, 5, 1)
        graph.addEdge(4, 8, 34)

        self.assertEquals(graph.getEdge(2, 5), 1)
        self.assertEquals(graph.getEdge(5, 2), 1)
        self.assertEquals(graph.getEdge(4, 8), 34)
        self.assertEquals(graph.getEdge(8, 4), 34)

        self.assertEquals(graph.getEdge(4, 4), None)

    def testGetVertex(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.setVertex(1, numpy.array([4, 5, 2]))

        self.assertRaises(ValueError, graph.setVertex, -1, numpy.array([4, 5, 2]))
        self.assertRaises(ValueError, graph.setVertex, 11, numpy.array([4, 5, 2]))
        self.assertRaises(ValueError, graph.setVertex, 2, numpy.array([4, 5, 2, 8]))
        self.assertRaises(ValueError, graph.setVertex, 2, numpy.array([4, 5]))
        self.assertTrue((graph.getVertex(1) == numpy.array([4, 5, 2])).all())
        self.assertTrue((graph.getVertex(0) == numpy.array([0, 0, 0])).all())

    def testSetVertex(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.setVertex(1, numpy.array([4, 5, 2]))

        self.assertTrue((graph.getVertex(1) == numpy.array([4, 5, 2])).all())
        self.assertTrue((graph.getVertex(0) == numpy.array([0, 0, 0])).all())

        graph.setVertex(1, numpy.array([8, 3, 1]))
        self.assertTrue((graph.getVertex(1) == numpy.array([8, 3, 1])).all())

    def testIsUndirected(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        self.assertEquals(graph.isUndirected(), True)

        graph = self.GraphType(vList, False)
        self.assertEquals(graph.isUndirected(), False)

    def testGetAllVertexIds(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        self.assertTrue((graph.getAllVertexIds() == numpy.array(list(range(0, numVertices)))).all())

    def testSubgraph(self):
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        vertices = numpy.random.rand(numVertices, numFeatures)
        vList.setVertices(vertices)

        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(2, 1)
        graph.addEdge(2, 5)
        graph.addEdge(2, 6)
        graph.addEdge(6, 9)

        subgraph = graph.subgraph([0,1,2,3])

        self.assertEquals(subgraph.getNumVertices(), 4)
        self.assertEquals(subgraph.getVertexList().getNumFeatures(), numFeatures)
        self.assertTrue((subgraph.getVertexList().getVertices(list(range(0, 4))) == vertices[list(range(0,4)), :]).all())
        self.assertEquals(subgraph.getNumEdges(), 4)
        self.assertTrue(subgraph.getEdge(0, 1) == 1)
        self.assertTrue(subgraph.getEdge(0, 2) == 1)
        self.assertTrue(subgraph.getEdge(0, 3) == 1)
        self.assertTrue(subgraph.getEdge(2, 1) == 1)

        subgraph = graph.subgraph([1,2,5,6])
        self.assertEquals(subgraph.getNumVertices(), 4)
        self.assertEquals(subgraph.getVertexList().getNumFeatures(), numFeatures)
        self.assertEquals(subgraph.getNumEdges(), 3)
        self.assertTrue((subgraph.getVertexList().getVertices([0,1,2,3]) == vertices[[1,2,5,6], :]).all())
        self.assertTrue(subgraph.getEdge(0, 1) == 1)
        self.assertTrue(subgraph.getEdge(1, 2) == 1)
        self.assertTrue(subgraph.getEdge(1, 3) == 1)


        #Test case of directed graph
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        vertices = numpy.random.rand(numVertices, numFeatures)
        vList.setVertices(vertices)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(2, 1)
        graph.addEdge(2, 5)
        graph.addEdge(2, 6)
        graph.addEdge(6, 9)

        subgraph = graph.subgraph([0,1,2,3])

        self.assertEquals(subgraph.isUndirected(), False)
        self.assertEquals(subgraph.getNumVertices(), 4)
        self.assertEquals(subgraph.getVertexList().getNumFeatures(), numFeatures)
        self.assertTrue((subgraph.getVertexList().getVertices(list(range(0, 4))) == vertices[list(range(0,4)), :]).all())
        self.assertEquals(subgraph.getNumEdges(), 4)
        self.assertTrue(subgraph.getEdge(0, 1) == 1)
        self.assertTrue(subgraph.getEdge(0, 2) == 1)
        self.assertTrue(subgraph.getEdge(0, 3) == 1)
        self.assertTrue(subgraph.getEdge(2, 1) == 1)

        subgraph = graph.subgraph([1,2,5,6])
        self.assertEquals(subgraph.getNumVertices(), 4)
        self.assertEquals(subgraph.getVertexList().getNumFeatures(), numFeatures)
        self.assertEquals(subgraph.getNumEdges(), 3)
        self.assertTrue((subgraph.getVertexList().getVertices([0,1,2,3]) == vertices[[1,2,5,6], :]).all())
        self.assertTrue(subgraph.getEdge(1, 0) == 1)
        self.assertTrue(subgraph.getEdge(1, 2) == 1)
        self.assertTrue(subgraph.getEdge(1, 3) == 1)

        subgraph = graph.subgraph([])

    def testAdd(self):
        numVertices = 5
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(2, 1)

        graph2 = self.GraphType(vList, False)
        graph2.addEdge(3, 2)
        graph2.addEdge(0, 4)
        graph2.addEdge(1, 3)
        graph2.addEdge(2, 1)

        newGraph = graph.add(graph2)

        #Check old graph is the same
        self.assertEquals(graph.getEdge(0,1) , 1)
        self.assertEquals(graph.getEdge(0,2) , 1)
        self.assertEquals(graph.getEdge(0,3) , 1)
        self.assertEquals(graph.getEdge(2,1) , 1)

        self.assertEquals(newGraph.getEdge(0,1) , 1)
        self.assertEquals(newGraph.getEdge(0,2) , 1)
        self.assertEquals(newGraph.getEdge(3,2) , 1)
        self.assertEquals(newGraph.getEdge(2,1) , 2)

        #Test edge addition of different sized graphs
        vList2 = VertexList(numVertices-1, numFeatures)
        graph2 = self.GraphType(vList2, False)
        graph2.addEdge(3, 2)

        self.assertRaises(ValueError, graph.add, graph2)

    def testMultiply(self):
        numVertices = 5
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(2, 1, 2)

        graph2 = self.GraphType(vList, False)
        graph2.addEdge(3, 2)
        graph2.addEdge(0, 4)
        graph2.addEdge(1, 3)
        graph2.addEdge(2, 1, 3)

        newGraph = graph.multiply(graph2)

        #Test old graph is the same
        self.assertEquals(graph.getEdge(0,1) , 1)
        self.assertEquals(graph.getEdge(0,2) , 1)
        self.assertEquals(graph.getEdge(0,3) , 1)
        self.assertEquals(graph.getEdge(2,1) , 2)

        self.assertEquals(newGraph.getNumEdges() , 1)
        self.assertEquals(newGraph.getEdge(0,1) , None)
        self.assertEquals(newGraph.getEdge(0,2) , None)
        self.assertEquals(newGraph.getEdge(3,2) , None)
        self.assertEquals(newGraph.getEdge(2,1) , 6)

        #Test edge multiplication of different sized graphs
        vList2 = VertexList(numVertices-1, numFeatures)
        graph2 = self.GraphType(vList2, False)
        graph2.addEdge(3, 2)

        self.assertRaises(ValueError, graph.multiply, graph2)

    def testCopy(self):
        numVertices = 5
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(2, 1)

        graph2 = graph.copy()

        graph2.addEdge(3, 4)

        self.assertEquals(graph2.getEdge(3, 4), 1)
        self.assertEquals(graph.getEdge(3, 4), None)

    def testDensity(self):
        numVertices = 5
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, False)

        self.assertEquals(graph.density(), 0)

        graph.addEdge(3, 4)
        self.assertEquals(graph.density(), float(1)/20)

        graph = self.GraphType(vList, True)

        self.assertEquals(graph.density(), 0)

        graph.addEdge(3, 4)
        self.assertEquals(graph.density(), float(1)/10)

    def testDepthFirstSearch(self):
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)
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
        self.assertEquals(graph.depthFirstSearch(7), [7])
        

    def testBreadthFirstSearch(self):
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)
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
        

    def testDiameter(self):
        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)

        self.assertEquals(graph.diameter(), 2)

        graph.addEdge(3, 2)

        self.assertEquals(graph.diameter(), 2)

        graph.addEdge(3, 4)

        self.assertEquals(graph.diameter(), 3)

        graph.addEdge(4, 5)

        self.assertEquals(graph.diameter(), 4)

        graph.addEdge(0, 5)

        self.assertEquals(graph.diameter(), 3)

        P = graph.floydWarshall(False)
        self.assertEquals(graph.diameter(P=P), 3)

        #Now try directed graphs
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)

        self.assertEquals(graph.diameter(), 2)

        graph.addEdge(4, 3)

        self.assertEquals(graph.diameter(), 2)

        graph.addEdge(5, 4)
        graph.addEdge(6, 5)

        self.assertEquals(graph.diameter(), 3)

        graph.addEdge(6, 6)
        self.assertEquals(graph.diameter(), 3)

        P = graph.floydWarshall(False)
        self.assertEquals(graph.diameter(P=P), 3)

        #Test on graph with no edges

        graph = self.GraphType(vList, False)
        self.assertEquals(graph.diameter(), 0)

        #Now, test graphs with weights
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.5)
        graph.addEdge(1, 3, 0.9)

        self.assertAlmostEqual(graph.diameter(True), 1.4, places=7)

        P = graph.floydWarshall(True)
        self.assertAlmostEquals(graph.diameter(True, P=P), 1.4, places=7)

    def testEffectiveDiameter(self):
        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        graph.addEdge(1, 4)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)

        self.assertEquals(graph.diameter(), 2)
        self.assertEquals(graph.effectiveDiameter(1.0), 2)
        self.assertEquals(graph.effectiveDiameter(0.5), 2)

        P = graph.floydWarshall(False)
        self.assertEquals(graph.effectiveDiameter(1.0, P=P), 2)
        self.assertEquals(graph.effectiveDiameter(0.5, P=P), 2)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(2, 3)
        graph.addEdge(4, 5)
        graph.addEdge(5, 6)
        graph.addEdge(7, 8)
        graph.addEdge(8, 9)

        self.assertEquals(graph.effectiveDiameter(1.0), 2)
        self.assertEquals(graph.effectiveDiameter(0.75), 1)
        self.assertEquals(graph.effectiveDiameter(0.5), 1)

        P = graph.floydWarshall(False)
        self.assertEquals(graph.effectiveDiameter(1.0, P=P), 2)
        self.assertEquals(graph.effectiveDiameter(0.75, P=P), 1)
        self.assertEquals(graph.effectiveDiameter(0.5, P=P), 1)

        #Test on a disconnected graph
        graph = self.GraphType(vList, True)
        self.assertEquals(graph.effectiveDiameter(1.0), 0)
        self.assertEquals(graph.effectiveDiameter(0.75), 0)
        self.assertEquals(graph.effectiveDiameter(0.5), 0)
        self.assertEquals(graph.effectiveDiameter(0.1), 0)

        P = graph.floydWarshall(False)
        self.assertEquals(graph.effectiveDiameter(1.0, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.75, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.5, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.1, P=P), 0)

        graph = self.GraphType(vList, False)
        self.assertEquals(graph.effectiveDiameter(1.0), 0)
        self.assertEquals(graph.effectiveDiameter(0.75), 0)
        self.assertEquals(graph.effectiveDiameter(0.5), 0)
        self.assertEquals(graph.effectiveDiameter(0.1), 0)

        P = graph.floydWarshall(False)
        self.assertEquals(graph.effectiveDiameter(1.0, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.75, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.5, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.1, P=P), 0)

        #Test on graph with 1 edge
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 0)
        self.assertEquals(graph.effectiveDiameter(1.0), 0)
        self.assertEquals(graph.effectiveDiameter(0.75), 0)
        self.assertEquals(graph.effectiveDiameter(0.5), 0)
        self.assertEquals(graph.effectiveDiameter(0.1), 0)

        P = graph.floydWarshall(False)
        self.assertEquals(graph.effectiveDiameter(1.0, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.75, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.5, P=P), 0)
        self.assertEquals(graph.effectiveDiameter(0.1, P=P), 0)

    def testFindComponents(self):
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)
        graph.addEdge(2, 6)
        graph.addEdge(4, 5)

        self.assertEquals(graph.findConnectedComponents()[0], [0,1,2,3,6])
        self.assertEquals(graph.findConnectedComponents()[1], [4, 5])

        graph = self.GraphType(vList, False)
        self.assertRaises(ValueError, graph.findConnectedComponents)

    #This doesn't seem to be a conclusive test
    def testFitPowerLaw(self):
        numVertices = 1000
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)

        ell = 2
        m = 2
        generator = BarabasiAlbertGenerator(ell, m)
        graph = generator.generate(graph)
        #logging.debug(graph.degreeDistribution())

        alpha, ks, xmin = graph.fitPowerLaw()
        self.assertAlmostEquals(alpha, 3.0, places=0)

    def testFloydWarshall(self):
        P = self.graph.floydWarshall()

        P2 = numpy.zeros((self.numVertices, self.numVertices))
        P2[0, :] = numpy.array([0, 1, 2, 2, 1, numpy.inf])
        P2[1, :] = numpy.array([1, 0, 3, 1, 2, numpy.inf])
        P2[2, :] = numpy.array([2, 3, 0, 4, 3, numpy.inf])
        P2[3, :] = numpy.array([2, 1, 4, 0, 1, numpy.inf])
        P2[4, :] = numpy.array([1, 2, 3, 1, 0, numpy.inf])
        P2[5, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0])

        self.assertTrue((P == P2).all())

        #Now test the directed graph
        P = self.graph2.floydWarshall()

        P2 = numpy.zeros((self.numVertices, self.numVertices))
        P2[0, :] = numpy.array([0, 1, 2, 2, 1, numpy.inf])
        P2[1, :] = numpy.array([numpy.inf, 0, numpy.inf, 1, 2, numpy.inf])
        P2[2, :] = numpy.array([numpy.inf, numpy.inf, 0, 5, 6, numpy.inf])
        P2[3, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, 0, 1, numpy.inf])
        P2[4, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0, numpy.inf])
        P2[5, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0])

        self.assertTrue((P == P2).all())

    def testFindAllDistances(self):
        P = self.graph.findAllDistances()

        P2 = numpy.zeros((self.numVertices, self.numVertices))
        P2[0, :] = numpy.array([0, 1, 2, 2, 1, numpy.inf])
        P2[1, :] = numpy.array([1, 0, 3, 1, 2, numpy.inf])
        P2[2, :] = numpy.array([2, 3, 0, 4, 3, numpy.inf])
        P2[3, :] = numpy.array([2, 1, 4, 0, 1, numpy.inf])
        P2[4, :] = numpy.array([1, 2, 3, 1, 0, numpy.inf])
        P2[5, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0])

        self.assertTrue((P == P2).all())

        #Now test the directed graph
        P = self.graph2.findAllDistances()

        P2 = numpy.zeros((self.numVertices, self.numVertices))
        P2[0, :] = numpy.array([0, 1, 2, 2, 1, numpy.inf])
        P2[1, :] = numpy.array([numpy.inf, 0, numpy.inf, 1, 2, numpy.inf])
        P2[2, :] = numpy.array([numpy.inf, numpy.inf, 0, 5, 6, numpy.inf])
        P2[3, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, 0, 1, numpy.inf])
        P2[4, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0, numpy.inf])
        P2[5, :] = numpy.array([numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, 0])

        self.assertTrue((P == P2).all())

    def testEgoGraph(self):
        numVertices = 6
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(2, 1)
        graph.addEdge(2, 3)
        graph.addEdge(4, 1)

        egoGraph = graph.egoGraph(0)

        self.assertTrue(egoGraph.getNumVertices() == 4)
        self.assertTrue(egoGraph.getNumEdges() == 5)
        self.assertEquals(egoGraph.getEdge(0,1), 1)
        self.assertEquals(egoGraph.getEdge(0,2), 1)
        self.assertEquals(egoGraph.getEdge(0,3), 1)
        self.assertEquals(egoGraph.getEdge(2,1), 1)
        self.assertEquals(egoGraph.getEdge(2,3), 1)

        egoGraph = graph.egoGraph(4)

        self.assertTrue(egoGraph.getNumVertices() == 2)
        self.assertTrue(egoGraph.getNumEdges() == 1)
        self.assertEquals(egoGraph.getEdge(1,0), 1)

        egoGraph = graph.egoGraph(3)

        self.assertTrue(egoGraph.getNumVertices() == 3)
        self.assertTrue(egoGraph.getNumEdges() == 3)
        self.assertEquals(egoGraph.getEdge(0,2), 1)
        self.assertEquals(egoGraph.getEdge(0,1), 1)
        self.assertEquals(egoGraph.getEdge(2,1), 1)


        egoGraph = graph.egoGraph(5)
        self.assertTrue(egoGraph.getNumVertices() == 1)
        self.assertTrue(egoGraph.getNumEdges() == 0)

    def testStr(self):
        logging.debug((self.graph))

    def testRemoveAllEdges(self):
        numVertices = 6
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(2, 1)
        graph.addEdge(2, 3)
        graph.addEdge(4, 1)

        self.assertEquals(graph.getNumEdges(), 6)

        graph.removeAllEdges()
        self.assertTrue(graph.getEdge(0,1) == None)
        self.assertEquals(graph.getNumEdges(), 0)

    def testAdjacencyMatrix(self):
        numVertices = 3
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1, 0.5)
        graph.addEdge(2, 1, 0.2)
        graph.addEdge(1, 1, 0.1)

        A = graph.adjacencyMatrix()
        W = graph.getWeightMatrix()


        W2 = numpy.zeros((numVertices, numVertices))
        A2 = numpy.zeros((numVertices, numVertices))

        W2[0,1]= 0.5
        W2[2,1]= 0.2
        W2[1,1]= 0.1

        A2[0,1]= 1
        A2[2,1]= 1
        A2[1,1]= 1

        self.assertTrue((W == W2).all())
        self.assertTrue((A == A2).all())

    def testComplement(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        graph3 = graph.complement()

        self.assertTrue(graph3.isUndirected())
        self.assertEquals(graph3.getNumEdges(), (numVertices**2 + numVertices)/2)

        graph.addEdge(0, 1, 0.1)
        graph.addEdge(2, 1, 0.2)
        graph.addEdge(4, 2, 0.5)
        graph.addEdge(6, 7, 0.9)
        graph.addEdge(3, 3, 1.1)

        graph2 = graph.complement()

        self.assertTrue(graph2.isUndirected())
        self.assertEquals(graph2.getEdge(0, 1), None)
        self.assertEquals(graph2.getEdge(2, 1), None)
        self.assertEquals(graph2.getEdge(4, 2), None)
        self.assertEquals(graph2.getEdge(6, 7), None)
        self.assertEquals(graph2.getEdge(3, 3), None)

        self.assertEquals(graph2.getEdge(0,0), 1)
        self.assertEquals(graph2.getNumEdges(), 50)

        #Now test on directed graphs
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)

        graph3 = graph.complement()
        self.assertEquals(graph3.getNumEdges(), numVertices**2)

        graph.addEdge(0, 1, 0.1)
        graph.addEdge(2, 1, 0.2)
        graph.addEdge(4, 2, 0.5)
        graph.addEdge(6, 7, 0.9)
        graph.addEdge(3, 3, 1.1)

        graph2 = graph.complement()

        self.assertFalse(graph2.isUndirected())
        self.assertEquals(graph2.getEdge(0, 1), None)
        self.assertEquals(graph2.getEdge(2, 1), None)
        self.assertEquals(graph2.getEdge(4, 2), None)
        self.assertEquals(graph2.getEdge(6, 7), None)
        self.assertEquals(graph2.getEdge(3, 3), None)

        self.assertEquals(graph2.getEdge(0,0), 1)
        self.assertEquals(graph2.getEdge(1,0), 1)
        self.assertEquals(graph2.getNumEdges(), 95)

    def testFindTrees(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)

        graph.addEdge(0, 1, 1)
        graph.addEdge(0, 2, 1)
        graph.addEdge(1, 3, 1)
        graph.addEdge(4, 5, 1)
        graph.addEdge(6, 7, 1)

        trees = graph.findTrees()

        self.assertEquals(trees[0], [0,1,2,3])
        self.assertEquals(trees[1], [6,7])
        self.assertEquals(trees[2], [4,5])
        self.assertEquals(trees[3], [9])
        self.assertEquals(trees[4], [8])

        #Make sure the output tree sizes are in order
        graph = self.GraphType(vList, False)
        graph.addEdge(1, 2, 1)
        graph.addEdge(3, 4, 1)
        graph.addEdge(3, 5, 1)

        graph.addEdge(6, 7, 1)
        graph.addEdge(6, 8, 1)
        graph.addEdge(8, 9, 1)

        trees = graph.findTrees()
        self.assertEquals(set(trees[0]), set([6,7,8,9]))
        self.assertEquals(trees[1], [3,4,5])
        self.assertEquals(trees[2], [1,2])
        self.assertEquals(trees[3], [0])

        #Test on size 1 graph
        numVertices = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)

        trees = graph.findTrees()
        self.assertEquals([len(x) for x in trees], [1])

    def testSetWeightMatrix(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = self.GraphType(vList)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)

        W = numpy.zeros((numVertices, numVertices))
        W[1, 1] = 1
        W[2, 1] = 1
        W[1, 2] = 1

        graph.setWeightMatrix(W)
        self.assertTrue((graph.getAllEdges() == numpy.array([[1, 1], [2, 1]])).all())

        W[1, 3] = 1
        self.assertRaises(ValueError, graph.setWeightMatrix, W)
        W = numpy.zeros((numVertices, numVertices+1))
        self.assertRaises(ValueError, graph.setWeightMatrix, W)

        #Now, see if it works for undirected graphs
        graph = self.GraphType(vList, False)
        W = numpy.zeros((numVertices, numVertices))
        W[1, 0] = 1
        W[3, 1] = 1
        W[1, 3] = 1

        graph.setWeightMatrix(W)
        self.assertTrue((graph.getAllEdges() == numpy.array([[1, 0], [1,3], [3, 1]])).all())

    def testGetNumDirEdges(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.1)

        self.assertTrue(graph.getNumDirEdges() == 4)
        graph.addEdge(1, 1)
        self.assertTrue(graph.getNumDirEdges() == 5)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)

        self.assertTrue(graph.getNumDirEdges() == 2)
        graph.addEdge(1, 1)
        self.assertTrue(graph.getNumDirEdges() == 3)

    def testOutDegreeSequence(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.2)
        graph.addEdge(1, 5)

        self.assertTrue((graph.outDegreeSequence() == numpy.array([1, 3, 1, 0,0,1,0,0,0,0])).all() )

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 5)
        graph.addEdge(3, 3)

        self.assertTrue((graph.outDegreeSequence() == numpy.array([1, 2, 0, 1,0,0,0,0,0,0])).all() )

    def testInDegreeSequence(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 5)

        self.assertTrue((graph.inDegreeSequence() == numpy.array([1, 3, 1, 0,0,1,0,0,0,0])).all() )

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.2)
        graph.addEdge(1, 5)
        graph.addEdge(2, 1)
        graph.addEdge(3, 3)

        self.assertTrue((graph.inDegreeSequence() == numpy.array([0, 2, 1, 1,0,1,0,0,0,0])).all() )

    def testInDegreeDistribution(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        self.assertTrue((graph.inDegreeDistribution() == numpy.array([])).all())

        graph.addEdge(0, 1, 2)
        graph.addEdge(0, 2, 2)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(2, 4, 2)

        self.assertTrue((graph.inDegreeDistribution() == numpy.array([0, 2, 2, 0, 1])).all())

        #Try empty graph
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)

        self.assertTrue((graph.inDegreeDistribution() == numpy.array([5])).all())

        #Try a star like graph
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 0, 2)
        graph.addEdge(0, 1, 2)
        graph.addEdge(0, 2, 2)
        graph.addEdge(0, 3, 2)
        graph.addEdge(0, 4, 2)

        self.assertTrue((graph.inDegreeDistribution() == numpy.array([0, 4, 0, 0, 0, 1])).all())

        #Ought to try a directed graph 
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)

        self.assertTrue((graph.inDegreeDistribution() == numpy.array([])).all())

        graph.addEdge(0, 1, 2)
        graph.addEdge(0, 2, 2)
        graph.addEdge(1, 2, 2)
        graph.addEdge(2, 3, 2)
        graph.addEdge(2, 4, 2)

        self.assertTrue((graph.inDegreeDistribution() == numpy.array([1, 3, 1])).all())

    def testGeneralVertexList(self):
        #Very brief test to make sure sparse graph works with general vertex lists
        numVertices = 10
        vList = GeneralVertexList(numVertices)
        vList.setVertex(0, "a")
        vList.setVertex(1, "b")
        vList.setVertex(5, "c")

        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 5)

    def testFromNetworkXGraph(self):
        try:
            import networkx
        except ImportError as error:
            logging.debug(error)
            return

        nxGraph = networkx.Graph()
        nxGraph.graph["VListType"] = GeneralVertexList
        #nxGraph.graph["numFeatures"] = 2
        #nxGraph.add_node(0)
        nxGraph.add_edge(0, 1)
        nxGraph.add_edge(1, 2)
        nxGraph.add_edge(1, 3)

        graph = self.GraphType.fromNetworkXGraph(nxGraph)

        self.assertTrue(graph.getNumVertices() == 4)
        self.assertTrue(graph.isUndirected() == True)
        self.assertTrue(graph.getNumEdges() == 3)
        self.assertTrue(graph.getEdge(0, 1) == 1)
        self.assertTrue(graph.getEdge(1, 2) == 1)
        self.assertTrue(graph.getEdge(1, 3) == 1)

        #Try directed graphs
        nxGraph = networkx.DiGraph()
        nxGraph.graph["VListType"] = GeneralVertexList
        #nxGraph.add_node(0)
        nxGraph.add_edge(0, 1)
        nxGraph.add_edge(1, 2)
        nxGraph.add_edge(1, 3)

        graph = self.GraphType.fromNetworkXGraph(nxGraph)

        self.assertTrue(graph.getNumVertices() == 4)
        self.assertTrue(graph.isUndirected() == False)
        self.assertTrue(graph.getNumEdges() == 3)
        self.assertTrue(graph.getEdge(0, 1) == 1)
        self.assertTrue(graph.getEdge(1, 2) == 1)
        self.assertTrue(graph.getEdge(1, 3) == 1)

        #Using a multigraph should fail
        nxGraph = networkx.MultiGraph()
        self.assertRaises(ValueError, self.GraphType.fromNetworkXGraph, nxGraph)

        #Test node labels
        nxGraph = networkx.DiGraph()
        nxGraph.graph["VListType"] = GeneralVertexList
        nxGraph.add_node("a", label="abc")
        nxGraph.add_node("b", label="i")
        nxGraph.add_node("c", label="am")
        nxGraph.add_node("d", label="here")
        nxGraph.add_edge("a", "b")
        nxGraph.add_edge("b", "c")
        nxGraph.add_edge("b", "d")

        graph = self.GraphType.fromNetworkXGraph(nxGraph)

        nodeDict = {}
        for i in range(len(nxGraph.nodes())):
            nodeDict[nxGraph.nodes()[i]] = i

        self.assertTrue(graph.getNumVertices() == 4)
        self.assertTrue(graph.isUndirected() == False)
        self.assertTrue(graph.getNumEdges() == 3)
        self.assertTrue(graph.getEdge(nodeDict["a"], nodeDict["b"]) == 1)
        self.assertTrue(graph.getEdge(nodeDict["b"], nodeDict["c"]) == 1)
        self.assertTrue(graph.getEdge(nodeDict["b"], nodeDict["d"]) == 1)

        self.assertTrue(graph.getVertex(0) == "abc")
        self.assertTrue(graph.getVertex(1) == "am")
        self.assertTrue(graph.getVertex(2) == "i")
        self.assertTrue(graph.getVertex(3) == "here")

        #Test in conjunction with toNetworkXGraph
        numVertices = 10
        numFeatures = 2 
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))

        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(0, 5)
        graph.addEdge(2, 5)
        graph.addEdge(3, 4)

        nxGraph = graph.toNetworkXGraph()
        graph2 = self.GraphType.fromNetworkXGraph(nxGraph)

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(graph.getVertexList().getVertices(list(range(numVertices))) -graph2.getVertexList().getVertices(list(range(numVertices)))) < tol)
        self.assertEquals(graph.getNumEdges(), graph2.getNumEdges())
        
        for i in range(numVertices):
            for j in range(numVertices):
                self.assertEquals(graph.getEdge(i, j), graph2.getEdge(i, j))

        #Use a GeneralVertexList
        numVertices = 10
        vList = GeneralVertexList(numVertices)
        for i in range(numVertices):
            vList.setVertex(i, "s" + str(i))

        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(0, 5)
        graph.addEdge(2, 5)
        graph.addEdge(3, 4)

        nxGraph = graph.toNetworkXGraph()
        graph2 = self.GraphType.fromNetworkXGraph(nxGraph)

        for i in range(numVertices):
            self.assertEquals(graph.getVertex(i), graph2.getVertex(i))
            
        self.assertEquals(graph.getNumEdges(), graph2.getNumEdges())
        
        for i in range(numVertices):
            for j in range(numVertices):
                self.assertEquals(graph.getEdge(i, j), graph2.getEdge(i, j))


    def testDiameter2(self):
        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)


        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)

        self.assertEquals(graph.diameter2(), 2)

        graph.addEdge(3, 2)

        self.assertEquals(graph.diameter2(), 2)

        graph.addEdge(3, 4)

        self.assertEquals(graph.diameter2(), 3)

        graph.addEdge(4, 5)

        self.assertEquals(graph.diameter2(), 4)

        graph.addEdge(0, 5)

        self.assertEquals(graph.diameter2(), 3)


        #Now try directed graphs
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)

        self.assertEquals(graph.diameter2(), 2)

        graph.addEdge(4, 3)

        self.assertEquals(graph.diameter2(), 2)

        graph.addEdge(5, 4)
        graph.addEdge(6, 5)

        self.assertEquals(graph.diameter2(), 3)


        graph.addEdge(6, 6)
        self.assertEquals(graph.diameter2(), 3)

        #Test on graph with no edges

        graph = self.GraphType(vList, False)
        self.assertEquals(graph.diameter2(), 0)

    def testLaplacianMatrix(self):
        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)

        L = numpy.zeros((numVertices, numVertices))
        A = graph.adjacencyMatrix()

        for i in range(numVertices):
            for j in range(numVertices):
                if i == j:
                    L[i, j] = numpy.sum(A[i, :])
                elif A[i, j] != 0:
                    L[i, j] = -1
                else:
                    L[i, j] = 0

        self.assertTrue((L == graph.laplacianMatrix() ).all())

    def testLoad(self):
        try:
            numVertices = 10
            numFeatures = 1
            vList = VertexList(numVertices, numFeatures)
            vList.setVertices(numpy.random.rand(numVertices, numFeatures))

            graph = self.GraphType(vList, True)
            graph.addEdge(0, 1, 0.1)
            graph.addEdge(1, 2, 0.2)
            graph.addEdge(1, 3, 0.3)

            tempDir = PathDefaults.getTempDir()
            tempFile = tempDir + "testGraph"

            graph.save(tempFile)

            dataDir = PathDefaults.getDataDir()
            os.chdir(dataDir)
            currentPath = os.getcwd()
            graph2 = self.GraphType.load(tempFile)

            #Make sure save doesn't change the path
            self.assertEquals(os.getcwd(), currentPath)

            self.assertEquals(graph.getNumVertices(), graph.getNumVertices())
            self.assertEquals(graph.getNumEdges(), graph.getNumEdges())
            self.assertTrue(graph2.isUndirected() == True)
            self.assertTrue((graph.getVertexList().getVertices(list(range(numVertices))) == graph2.getVertexList().getVertices(list(range(numVertices)))).all())
            self.assertTrue((graph.getAllEdges() == graph2.getAllEdges()).all())
            self.assertTrue(graph2.getEdge(0, 1) == 0.1)
            self.assertTrue(graph2.getEdge(1, 2) == 0.2)
            self.assertTrue(graph2.getEdge(1, 3) == 0.3)

            #Test if loading of old-style graph files works
            testDir = PathDefaults.getDataDir() + "test/"
            graphFilename = testDir + "fd"

            graph = self.GraphType.load(graphFilename)

            self.assertEquals(graph.getEdge(1, 1), 1)
            self.assertEquals(graph.getEdge(2, 2), 1)
            self.assertEquals(graph.getNumVertices(), 10)
        except IOError as e:
            logging.warn(e)
            pass
        except OSError as e:
            logging.warn(e)
            pass

    def testMaxEigenvector(self):
        tol = 10**-6
        numVertices = 5
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2, 0.1)
        graph.addEdge(2, 0)

        v = graph.maxEigenvector()

        W = graph.getWeightMatrix()
        lmbda, U = numpy.linalg.eig(W)
        i = numpy.argmax(lmbda)

        self.assertTrue(numpy.linalg.norm(U[:, i] - v) < tol)

    def testMaxProductPaths(self):
        numVertices = 6
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 3, 0.1)
        graph.addEdge(0, 2, 0.2)
        graph.addEdge(2, 3, 0.5)
        graph.addEdge(0, 4, 0.1)
        graph.addEdge(3, 4, 0.1)

        P = graph.maxProductPaths()

        P2 = numpy.zeros((numVertices, numVertices))
        P2[0, :] = numpy.array([0.04, 0.1, 0.2, 0.1, 0.1, 0])
        P2[1, :] = numpy.array([0.1, 0.01, 0.05, 0.1, 0.01, 0])
        P2[2, :] = numpy.array([0.2, 0.05, 0.25, 0.5, 0.05, 0])
        P2[3, :] = numpy.array([0.1, 0.1, 0.5, 0.25, 0.1, 0])
        P2[4, :] = numpy.array([0.1, 0.01, 0.05, 0.1, 0.01, 0])
        P2[5, :] = numpy.array([0,0,0,0,0,0])

        self.assertAlmostEquals(numpy.linalg.norm(P - P2), 0, places=6)

        #Now test on a directed graph
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 3, 0.1)
        graph.addEdge(0, 2, 0.2)
        graph.addEdge(2, 3, 0.5)
        graph.addEdge(0, 4, 0.1)
        graph.addEdge(3, 4, 0.1)

        P = graph.maxProductPaths()

        P2 = numpy.zeros((numVertices, numVertices))
        P2[0, :] = numpy.array([0, 0.1, 0.2, 0.1, 0.1, 0])
        P2[1, :] = numpy.array([0, 0, 0, 0.1, 0.01, 0])
        P2[2, :] = numpy.array([0, 0, 0, 0.5, 0.05, 0])
        P2[3, :] = numpy.array([0, 0, 0, 0, 0.1, 0])
        P2[4, :] = numpy.array([0,0,0,0,0,0])
        P2[5, :] = numpy.array([0,0,0,0,0,0])

        self.assertAlmostEquals(numpy.linalg.norm(P - P2), 0, places=6)

    def testMaybeIsomorphicWith(self):
        numVertices = 6
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 3, 0.1)
        graph.addEdge(0, 2, 0.2)
        graph.addEdge(2, 3, 0.5)
        graph.addEdge(0, 4, 0.1)
        graph.addEdge(3, 4, 0.1)

        graph2 = self.GraphType(vList, True)
        graph2.addEdge(0, 1, 0.1)
        graph2.addEdge(1, 3, 0.1)
        graph2.addEdge(0, 2, 0.2)
        graph2.addEdge(2, 3, 0.5)
        graph2.addEdge(0, 4, 0.1)
        graph2.addEdge(3, 4, 0.1)
        graph2.addEdge(4, 5, 0.1)

        graph3 = self.GraphType(vList, True)
        graph3.addEdge(2, 4, 0.1)
        graph3.addEdge(4, 5, 0.1)
        graph3.addEdge(2, 1, 0.2)
        graph3.addEdge(1, 5, 0.5)
        graph3.addEdge(2, 0, 0.1)
        graph3.addEdge(5, 0, 0.1)

        self.assertTrue(graph.maybeIsomorphicWith(graph))
        self.assertFalse(graph.maybeIsomorphicWith(graph2))
        self.assertTrue(graph.maybeIsomorphicWith(graph3))

    def testSave(self):
        try:
            numVertices = 10
            numFeatures = 1
            vList = VertexList(numVertices, numFeatures)
            vList.setVertices(numpy.random.rand(numVertices, numFeatures))

            graph = self.GraphType(vList, False)
            graph.addEdge(0, 1, 0.1)
            graph.addEdge(1, 2, 0.2)
            graph.addEdge(1, 3, 0.3)

            dataDir = PathDefaults.getDataDir()
            os.chdir(dataDir)

            tempDir = PathDefaults.getTempDir()
            currentPath = os.getcwd()
            graph.save(tempDir + "testGraph")

            #Make sure save doesn't change the path
            self.assertEquals(os.getcwd(), currentPath)
        except IOError as e:
            logging.warn(e)
            pass
        except OSError as e:
            logging.warn(e)
            pass

    def testSetVertices(self):
        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))

        graph = self.GraphType(vList, False)

        X = numpy.random.rand(numVertices, numFeatures)

        vertexIndices =list(range(0, numVertices))
        graph.setVertices(vertexIndices, X)

        vertexIndices2 = graph.getAllVertexIds()
        vertices2 = graph.getVertices(vertexIndices2)

        self.assertEquals(vertexIndices, vertexIndices2)
        self.assertTrue((X == vertices2).all())

    def testToNetworkXGraph(self):
        try:
            import networkx
        except ImportError as error:
            logging.debug(error)
            return

        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList)
        graph.addEdge(5, 1, 4)
        graph.addEdge(5, 2, 2)
        graph.addEdge(2, 7, 4)
        graph.addEdge(1, 9, 6)

        graph2 = self.GraphType(vList, False)
        graph2.addEdge(5, 1, 4)
        graph2.addEdge(5, 2, 2)
        graph2.addEdge(2, 7, 4)
        graph2.addEdge(1, 9, 6)
        networkXGraph = graph.toNetworkXGraph()

        self.assertEquals(networkXGraph.get_edge_data(5, 1), {'value' : 4.0})
        self.assertEquals(networkXGraph.get_edge_data(5, 2), {'value' : 2.0})
        self.assertEquals(networkXGraph.get_edge_data(2, 7), {'value' : 4.0})
        self.assertEquals(networkXGraph.get_edge_data(1, 9), {'value' : 6.0})
        self.assertEquals(networkXGraph.get_edge_data(9, 1), {'value' : 6.0})

        vertexIndexList = []

        for i in networkXGraph.__iter__():
            vertexIndexList.append(i)

        vertexIndexList.sort()
        self.assertTrue(vertexIndexList == list(range(numVertices)))
        self.assertTrue(networkXGraph.edges() == [(1, 9), (1, 5), (2, 5), (2, 7)])
        self.assertTrue(type(networkXGraph) == networkx.Graph)

        #Now we test the case where we have a directed graph
        networkXGraph = graph2.toNetworkXGraph()

        self.assertEquals(networkXGraph.get_edge_data(5, 1), {'value' : 4.0})
        self.assertEquals(networkXGraph.get_edge_data(5, 2), {'value' : 2.0})
        self.assertEquals(networkXGraph.get_edge_data(2, 7), {'value' : 4.0})
        self.assertEquals(networkXGraph.get_edge_data(1, 9), {'value' : 6.0})
        self.assertFalse(networkXGraph.has_edge(9, 1))

        vertexIndexList = []

        for i in networkXGraph.__iter__():
            vertexIndexList.append(i)

        vertexIndexList.sort()
        self.assertTrue(vertexIndexList == list(range(numVertices)))
        self.assertTrue(networkXGraph.edges() == [(1, 9), (2, 7), (5, 1), (5, 2)])
        self.assertTrue(type(networkXGraph) == networkx.DiGraph)

        #Test a graph with no edges
        numVertices = 10
        numFeatures = 3
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))

        graph = self.GraphType(vList)
        networkXGraph = graph.toNetworkXGraph()

        self.assertTrue(networkXGraph.order() == numVertices)
        self.assertTrue(networkXGraph.size() == 0)

        self.assertTrue((networkXGraph.nodes(data=True)[0][1]['label'] ==graph.getVertex(0)).all())

    def testTriangleSequence(self):
        tol = 10**-6
        numVertices = 5
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, True)

        seq = graph.triangleSequence()
        self.assertTrue(numpy.linalg.norm(seq - numpy.array([0, 0, 0, 0, 0])) < tol)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2, 0.1)
        graph.addEdge(1, 2)

        seq = graph.triangleSequence()
        self.assertTrue(numpy.linalg.norm(seq - numpy.array([2, 2, 2, 0, 0])) < tol)

        graph.addEdge(2, 3)
        graph.addEdge(3, 0, -0.3)

        seq = graph.triangleSequence()
        self.assertTrue(numpy.linalg.norm(seq - numpy.array([4, 2, 4, 2, 0])) < tol)

        graph.removeAllEdges()
        graph.addEdge(0, 0)
        seq = graph.triangleSequence()
        self.assertTrue(numpy.linalg.norm(seq - numpy.array([0, 0, 0, 0, 0])) < tol)

        #Test on directed graphs
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2, 0.1)
        graph.addEdge(2, 0)

        seq = graph.triangleSequence()
        self.assertTrue(numpy.linalg.norm(seq - numpy.array([1, 1, 1, 0, 0])) < tol)

        graph.addEdge(0, 3)
        graph.addEdge(3, 4, 0.1)
        graph.addEdge(4, 0)

        seq = graph.triangleSequence()
        self.assertTrue(numpy.linalg.norm(seq - numpy.array([2, 1, 1, 1, 1])) < tol)

    def testUnion(self):
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(5, 2, 0.1)
        graph.addEdge(6, 0, 0.1)

        graph2 = self.GraphType(vList, True)
        graph2.addEdge(0, 2, 0.1)
        graph2.addEdge(5, 3, 0.1)
        graph2.addEdge(5, 2, 0.1)

        newGraph = graph.union(graph2)

        #Test original graph is the same
        self.assertEquals(graph.getEdge(0, 1), 0.1)
        self.assertEquals(graph.getEdge(5, 2), 0.1)
        self.assertEquals(graph.getEdge(6, 0), 0.1)

        self.assertEquals(newGraph.getNumEdges(), 5)
        self.assertEquals(newGraph.getEdge(0, 1), 1)
        self.assertEquals(newGraph.getEdge(5, 2), 1)
        self.assertEquals(newGraph.getEdge(6, 0), 1)
        self.assertEquals(newGraph.getEdge(0, 2), 1)
        self.assertEquals(newGraph.getEdge(5, 3), 1)

        #Test union of graph 2 with itself
        newGraph = graph2.union(graph2)
        self.assertEquals(newGraph.getNumEdges(), 3)
        self.assertEquals(newGraph.getEdge(0, 2), 1)
        self.assertEquals(newGraph.getEdge(5, 3), 1)
        self.assertEquals(newGraph.getEdge(5, 2), 1)

    def testIntersect(self):
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(5, 2, 0.1)
        graph.addEdge(6, 0, 0.1)

        graph2 = self.GraphType(vList, True)
        graph2.addEdge(0, 2, 0.1)
        graph2.addEdge(5, 3, 0.1)
        graph2.addEdge(5, 2, 0.1)

        newGraph = graph.intersect(graph2)

        #Test old graph is the same
        self.assertEquals(graph.getEdge(0, 1), 0.1)
        self.assertEquals(graph.getEdge(5, 2), 0.1)
        self.assertEquals(graph.getEdge(6, 0), 0.1)

        self.assertEquals(newGraph.getNumEdges(), 1)
        self.assertEquals(newGraph.getEdge(5, 2), 1)

        #Test intersect of graph 2 with itself
        newGraph = graph2.intersect(graph2)
        self.assertEquals(newGraph.getNumEdges(), 3)
        self.assertEquals(newGraph.getEdge(0, 2), 1)
        self.assertEquals(newGraph.getEdge(5, 3), 1)
        self.assertEquals(newGraph.getEdge(5, 2), 1)

    def testIsTree(self):
        numVertices = 3
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)

        graph.addEdge(0, 1)
        self.assertFalse(graph.isTree())

        graph.addEdge(0, 2)
        self.assertTrue(graph.isTree())

        graph.addEdge(2, 0)
        self.assertFalse(graph.isTree())

        graph = self.GraphType(vList, True)
        self.assertRaises(ValueError, graph.isTree)

        #Try a bigger graph
        numVertices = 6
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(0, 4)
        graph.addEdge(0, 5)

        self.assertTrue(graph.isTree())

        graph.removeEdge(0, 5)
        graph.addEdge(1, 5)
        self.assertTrue(graph.isTree())

        #Try 1 node graph
        numVertices = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, False)
        self.assertTrue(graph.isTree())

    def testBetweenness(self):
        tol = 10**-6
        numVertices = 5
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)

        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1)
        graph.addEdge(1, 2, 0.1)
        graph.addEdge(2, 3, 0.1)
        graph.addEdge(0, 3, 0.1)

        #logging.debug(graph.betweenness())

    def testSetVertexList(self):
        numVertices = 5
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))

        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.2)

        self.assertTrue((graph.getVertex(0) == vList.getVertex(0)).all())
        self.assertTrue((graph.getVertex(1) == vList.getVertex(1)).all())
        self.assertTrue((graph.getVertex(2) == vList.getVertex(2)).all())

        vList2 = VertexList(numVertices, numFeatures+2)
        vList2.setVertices(numpy.random.rand(numVertices, numFeatures+2))

        graph.setVertexList(vList2)
        self.assertTrue((graph.getVertex(0) == vList2.getVertex(0)).all())
        self.assertTrue((graph.getVertex(1) == vList2.getVertex(1)).all())
        self.assertTrue((graph.getVertex(2) == vList2.getVertex(2)).all())

        vList3 = VertexList(numVertices+1, numFeatures)
        self.assertRaises(ValueError, graph.setVertexList, vList3)



    def testNormalisedLaplacianSym(self):
        numVertices = 10
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 9)
        graph.addEdge(1, 1)
        graph.addEdge(1, 5)

        L = graph.normalisedLaplacianSym()

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

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(L2 - L) < tol)

    def testNormalisedLaplacianRw(self):
        numVertices = 10
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 9)
        graph.addEdge(1, 1)
        graph.addEdge(1, 5)

        L = graph.normalisedLaplacianRw()

        W = graph.getWeightMatrix()
        L2 = numpy.zeros((numVertices, numVertices))
        d = graph.outDegreeSequence()

        for i in range(numVertices):
            for j in range(numVertices):
                if d[i] != 0 and d[j]!= 0:
                    Wij = W[i, j]/(d[i])
                else:
                    Wij = 0

                if i == j:
                    L2[i, j] = 1 - Wij
                else:
                    L2[i, j] = -Wij

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(L2 - L) < tol)

    def testSetDiff(self):
        numVertices = 10
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(5, 2, 0.1)
        graph.addEdge(6, 0, 0.1)
        graph.addEdge(6, 1, 0.1)

        graph2 = self.GraphType(vList, True)
        graph2.addEdge(0, 1, 0.1)
        graph2.addEdge(5, 3, 0.1)
        graph2.addEdge(5, 2, 0.1)

        newGraph = graph.setDiff(graph2)

        #Test old graph is the same
        self.assertEquals(graph.getEdge(0, 1), 0.1)
        self.assertEquals(graph.getEdge(5, 2), 0.1)
        self.assertEquals(graph.getEdge(6, 0), 0.1)
        self.assertEquals(graph.getEdge(6, 1), 0.1)

        self.assertEquals(newGraph.getNumEdges(), 2)
        self.assertEquals(newGraph.getEdge(6, 0), 1)
        self.assertEquals(newGraph.getEdge(6, 1), 1)

        #Test setdiff of graph 2 with itself
        newGraph = graph2.setDiff(graph2)
        self.assertEquals(newGraph.getNumEdges(), 0)

    def testIncidenceMatrix(self):
        numVertices = 5
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.1)
        graph.addEdge(3, 0, 0.1)
        graph.addEdge(4, 1, 0.1)

        X = graph.incidenceMatrix().todense()
        L = X.dot(X.T)
        L2 = graph.laplacianMatrix()

        #In the case of undirected graphs we get the laplacian 
        self.assertTrue((L==L2).all())

        #Directed graph - we get something different 
        graph = self.GraphType(vList, False)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.1)
        graph.addEdge(3, 0, 0.1)
        graph.addEdge(4, 1, 0.1)

        X = graph.incidenceMatrix().todense()
        L = X.dot(X.T)
        L2 = graph.laplacianMatrix()

    def testDegreeSequence(self):
        numVertices = 5
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.1)
        graph.addEdge(3, 0, 0.1)
        graph.addEdge(4, 1, 0.1)

        self.assertTrue((graph.degreeSequence() == [2, 3, 1, 1, 1]).all())

        #Now add a self edge
        graph.addEdge(0, 0)
        self.assertTrue((graph.degreeSequence() == [4, 3, 1, 1, 1]).all())

        graph.addEdge(1, 1)
        self.assertTrue((graph.degreeSequence() == [4, 5, 1, 1, 1]).all())

    def testAdjacencyList(self):
        numVertices = 5
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 2, 0.2)
        graph.addEdge(3, 0, 0.3)
        graph.addEdge(4, 1, 0.4)

        L, W = graph.adjacencyList()

        for i in range(numVertices):
            self.assertTrue((L[i]==numpy.sort(graph.neighbours(i))).all())

        self.assertTrue(W[0][0] == 0.1)
        self.assertTrue(W[0][1] == 0.3)
        self.assertTrue(W[4][0] == 0.4)

        #Now use just adjacencies 
        L, W = graph.adjacencyList(False)

        for i in range(numVertices):
            self.assertTrue((L[i]==numpy.sort(graph.neighbours(i))).all())

        self.assertTrue(W[0][0] == 1)
        self.assertTrue(W[0][1] == 1)
        self.assertTrue(W[4][0] == 1)

    def testGetItem(self):

        numVertices = 5
        graph = self.GraphType(GeneralVertexList(numVertices))
        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)
        graph.addEdge(2, 4, 1)
        graph.addEdge(2, 3, 2)
        graph.setVertex(0, "abc")

        self.assertEquals(graph[1,1], 0.1)
        self.assertEquals(graph[1,3], 0.5)


    def testSetItem(self):
        numVertices = 5
        graph = self.GraphType(GeneralVertexList(numVertices))
        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)

        self.assertEquals(graph[1,3], 0.5)
        graph[1, 3] = 2
        self.assertEquals(graph[1,3], 2)


    def testToIGraph(self):
        try:
            import igraph
        except ImportError as error:
            logging.debug(error)
            return

        numVertices = 7
        graph = self.GraphType(GeneralVertexList(numVertices))
        graph.addEdge(1, 1, 0.1)
        graph.addEdge(1, 3, 0.5)
        graph.addEdge(1, 5, 0.5)
        graph.addEdge(3, 5, 0.5)
        graph.addEdge(5, 6, 0.1)

        graph.setVertex(1, "a")
        graph.setVertex(2, "b")
        graph.setVertex(3, "c")

        igraph = graph.toIGraph()

        self.assertEquals(len(igraph.vs), graph.getNumVertices())
        self.assertEquals(len(igraph.es), graph.getNumEdges())

        self.assertEquals(igraph.vs["label"][1], "a")
        self.assertEquals(igraph.vs["label"][2], "b")
        self.assertEquals(igraph.vs["label"][3], "c")

        edges = igraph.get_edgelist()

        i = 0 
        for e in edges:
            self.assertTrue(graph.getEdge(e[0], e[1]) == igraph.es[i]["value"])
            i += 1 
            
    def testPickle(self): 
        numVertices = 10
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        graph = self.GraphType(vList)  
        graph[0, 0] = 1
        graph[3, 5] = 0.1
        graph.setVertex(0, numpy.array([12.3]))
        
        output = pickle.dumps(graph)
        newGraph = pickle.loads(output)
        
        graph[2, 2] = 1
        
        self.assertEquals(newGraph[0, 0], 1)
        self.assertEquals(newGraph[3, 5], 0.1)
        self.assertEquals(newGraph[2, 2], 0.0)
        self.assertEquals(newGraph.getNumEdges(), 2)
        self.assertEquals(newGraph.getNumVertices(), numVertices)
        self.assertEquals(newGraph.isUndirected(), True)
        
        self.assertEquals(graph[0, 0], 1)
        self.assertEquals(graph[3, 5], 0.1)
        self.assertEquals(graph[2, 2], 1)
        self.assertEquals(graph.getNumEdges(), 3)
        self.assertEquals(graph.getNumVertices(), numVertices)
        self.assertEquals(graph.isUndirected(), True)
        
        for i in range(numVertices): 
            nptst.assert_array_equal(graph.getVertex(i), newGraph.getVertex(i))
            
    def testToDictGraph(self): 
        dictGraph = self.graph.toDictGraph() 
        
        edges = self.graph.getAllEdges()
            
        for i in range(edges.shape[0]): 
            self.assertEquals(dictGraph[edges[i, 0], edges[i, 1]], self.graph[edges[i, 0], edges[i, 1]])
        
        dictGraph2 = self.graph2.toDictGraph() 
        
        edges2 = self.graph2.getAllEdges()
            
        for i in range(edges2.shape[0]): 
            self.assertEquals(dictGraph2[edges2[i, 0], edges2[i, 1]], self.graph[edges2[i, 0], edges2[i, 1]])
        