
from apgl.util import *
from apgl.graph.VertexList import VertexList
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.DenseGraph import DenseGraph
from apgl.graph.GraphUtils import GraphUtils
from apgl.generator.BarabasiAlbertGenerator import BarabasiAlbertGenerator
import unittest
import numpy
import scipy.sparse

class AbstractMatrixGraphTest(unittest.TestCase):
    def setUp(self):
        pass

    def testTreeDepth(self):
        numVertices = 4
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(2, 3)
        self.assertEquals(GraphUtils.treeDepth(graph), 2)

        numVertices = 5
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList, False)
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(2, 3)
        graph.addEdge(3, 4)
        self.assertEquals(GraphUtils.treeDepth(graph), 3)
        
    def testVertexLabelPairs(self):
        numVertices = 6
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.array([numpy.arange(0, 6)]).T)

        graph = DenseGraph(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 3, 0.1)
        graph.addEdge(0, 2, 0.2)
        graph.addEdge(2, 3, 0.5)
        graph.addEdge(0, 4, 0.1)
        graph.addEdge(3, 4, 0.1)

        tol = 10**-6
        edges = graph.getAllEdges()

        X = GraphUtils.vertexLabelPairs(graph, edges)
        self.assertTrue(numpy.linalg.norm(X - edges) < tol )


        X = GraphUtils.vertexLabelPairs(graph, edges[[5, 2, 1], :])
        self.assertTrue(numpy.linalg.norm(X - edges[[5,2,1], :]) < tol )

        #Try a bigger graph
        numVertices = 6
        numFeatures = 2
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.randn(numVertices, numFeatures))

        graph = DenseGraph(vList, True)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 3, 0.1)

        edges = graph.getAllEdges()

        X = GraphUtils.vertexLabelPairs(graph, edges)
        self.assertTrue(numpy.linalg.norm(X[0, 0:numFeatures] - vList.getVertex(1)) < tol )
        self.assertTrue(numpy.linalg.norm(X[0, numFeatures:numFeatures*2] - vList.getVertex(0)) < tol )
        self.assertTrue(numpy.linalg.norm(X[1, 0:numFeatures] - vList.getVertex(3)) < tol )
        self.assertTrue(numpy.linalg.norm(X[1, numFeatures:numFeatures*2] - vList.getVertex(1)) < tol )

        #Try directed graphs
        graph = DenseGraph(vList, False)
        graph.addEdge(0, 1, 0.1)
        graph.addEdge(1, 3, 0.1)

        edges = graph.getAllEdges()

        X = GraphUtils.vertexLabelPairs(graph, edges)
        self.assertTrue(numpy.linalg.norm(X[0, 0:numFeatures] - vList.getVertex(0)) < tol )
        self.assertTrue(numpy.linalg.norm(X[0, numFeatures:numFeatures*2] - vList.getVertex(1)) < tol )
        self.assertTrue(numpy.linalg.norm(X[1, 0:numFeatures] - vList.getVertex(1)) < tol )
        self.assertTrue(numpy.linalg.norm(X[1, numFeatures:numFeatures*2] - vList.getVertex(3)) < tol )

    def testModularity(self):
        numVertices = 6
        graph = SparseGraph(GeneralVertexList(numVertices))

        graph.addEdge(0,0)
        graph.addEdge(1,1)
        graph.addEdge(2,2)
        graph.addEdge(0,1)
        graph.addEdge(0,2)
        graph.addEdge(2,1)

        graph.addEdge(3,4,2)
        graph.addEdge(3,5,2)
        graph.addEdge(4,5,2)
        graph.addEdge(3,3,2)
        graph.addEdge(4,4,2)
        graph.addEdge(5,5,2)

        W = graph.getWeightMatrix()
        clustering = numpy.array([0,0,0,1,1,1])

        #This is the same as the igraph result
        Q = GraphUtils.modularity(W, clustering)
        self.assertEquals(Q, 4.0/9.0)

        Ws = scipy.sparse.csr_matrix(W)
        Q = GraphUtils.modularity(Ws, clustering)
        self.assertEquals(Q, 4.0/9.0)

        W = numpy.ones((numVertices, numVertices))
        Q = GraphUtils.modularity(W, clustering)

        self.assertEquals(Q, 0.0)

        Ws = scipy.sparse.csr_matrix(W)
        Q = GraphUtils.modularity(Ws, clustering)
        self.assertEquals(Q, 0.0)

    def testKwayNormalisedCut(self):
        numVertices = 6
        graph = SparseGraph(GeneralVertexList(numVertices))

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(2, 1)

        graph.addEdge(3, 4)
        graph.addEdge(3, 5)
        graph.addEdge(5, 4)

        W = graph.getWeightMatrix()
        clustering = numpy.array([0,0,0, 1,1,1])

        self.assertEquals(GraphUtils.kwayNormalisedCut(W, clustering), 0.0)

        #Try sparse W
        Ws = scipy.sparse.csr_matrix(W)
        self.assertEquals(GraphUtils.kwayNormalisedCut(Ws, clustering), 0.0)

        graph.addEdge(2, 3)
        W = graph.getWeightMatrix()
        self.assertEquals(GraphUtils.kwayNormalisedCut(W, clustering), 1.0/7)

        Ws = scipy.sparse.csr_matrix(W)
        self.assertEquals(GraphUtils.kwayNormalisedCut(Ws, clustering), 1.0/7)

        clustering = numpy.array([0,0,0, 1,1,2])
        self.assertEquals(GraphUtils.kwayNormalisedCut(W, clustering), 61.0/105)

        self.assertEquals(GraphUtils.kwayNormalisedCut(Ws, clustering), 61.0/105)

        #Test two vertices without any edges
        W = numpy.zeros((2, 2))
        clustering = numpy.array([0, 1])
        self.assertEquals(GraphUtils.kwayNormalisedCut(W, clustering), 0.0)

        Ws = scipy.sparse.csr_matrix(W)
        self.assertEquals(GraphUtils.kwayNormalisedCut(Ws, clustering), 0.0)

    def testShiftLaplacian(self):
        numVertices = 10
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        ell = 2
        m = 2
        generator = BarabasiAlbertGenerator(ell, m)
        graph = generator.generate(graph)

        k = 10
        W = graph.getSparseWeightMatrix()
        L = GraphUtils.shiftLaplacian(W)

        L2 = 2*numpy.eye(numVertices) - graph.normalisedLaplacianSym()

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(L - L2) < tol)


    def testRandIndex(self): 
        clustering1 = numpy.array([1, 1, 1, 2, 2, 2])
        clustering2 = numpy.array([2, 2, 2, 1, 1, 1])
        
        self.assertEquals(GraphUtils.randIndex(clustering1, clustering2), 0.0)
        
        clustering2 = numpy.array([2, 2, 2, 1, 1, 2])
        self.assertEquals(GraphUtils.randIndex(clustering1, clustering2), 1/3.0) 
        
        clustering2 = numpy.array([1, 2, 2, 1, 1, 2])
        self.assertEquals(GraphUtils.randIndex(clustering1, clustering2), 16/30.0) 

if __name__ == '__main__':
    unittest.main()