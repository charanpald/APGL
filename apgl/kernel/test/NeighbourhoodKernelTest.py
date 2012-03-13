
import unittest
from apgl.kernel.NeighbourhoodKernel import NeighbourhoodKernel
from apgl.graph.VertexList import VertexList
from apgl.graph.DenseGraph import DenseGraph
from apgl.graph.SparseGraph import SparseGraph


class  NeighbourhoodKernelTest(unittest.TestCase):

    def testComputeNeighbourhoodGraphs(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)
        graph.addEdge(0,3)
        graph.addEdge(0,1)
        graph.addEdge(2,1)
        graph.addEdge(3,4)
        graph.addEdge(1,9)
        graph.addEdge(9,7)
        graph.addEdge(6, 8)

        r = 2 

        kernel = NeighbourhoodKernel(r)
        subGraphs = kernel.computeNeighbourhoodGraphs(graph)

        self.assertTrue(subGraphs[0] == set([0,1,2,3,4,9]))
        self.assertTrue(subGraphs[1] == set([0,1,2,3,7,9]))
        self.assertTrue(subGraphs[2] == set([0,1,2,9]))
        self.assertTrue(subGraphs[3] == set([0,1,3,4]))
        self.assertTrue(subGraphs[4] == set([0,3,4]))
        self.assertTrue(subGraphs[5] == set([5]))
        self.assertTrue(subGraphs[6] == set([6,8]))
        self.assertTrue(subGraphs[7] == set([1,7,9]))
        self.assertTrue(subGraphs[8] == set([6,8]))
        self.assertTrue(subGraphs[9] == set([0, 1,2,7,9]))

        r = 1
        kernel = NeighbourhoodKernel(r)
        subGraphs = kernel.computeNeighbourhoodGraphs(graph)

        self.assertTrue(subGraphs[0] == set([0,1,3]))
        self.assertTrue(subGraphs[1] == set([0,1,2,9]))
        self.assertTrue(subGraphs[2] == set([1,2]))
        self.assertTrue(subGraphs[3] == set([0,3,4]))
        self.assertTrue(subGraphs[4] == set([3,4]))
        self.assertTrue(subGraphs[5] == set([5]))
        self.assertTrue(subGraphs[6] == set([6,8]))
        self.assertTrue(subGraphs[7] == set([7,9]))
        self.assertTrue(subGraphs[8] == set([6,8]))
        self.assertTrue(subGraphs[9] == set([1,7,9]))


        #Test directed graphs
        graph = SparseGraph(vList, False)
        graph.addEdge(0,3)
        graph.addEdge(0,1)
        graph.addEdge(2,1)
        graph.addEdge(3,4)
        graph.addEdge(1,9)
        graph.addEdge(9,7)
        graph.addEdge(6,8)

        r = 2
        kernel = NeighbourhoodKernel(r)
        subGraphs = kernel.computeNeighbourhoodGraphs(graph)

        self.assertTrue(subGraphs[0] == set([0,1,3,4,9]))
        self.assertTrue(subGraphs[1] == set([1, 9, 7]))
        self.assertTrue(subGraphs[2] == set([1,2,9]))
        self.assertTrue(subGraphs[3] == set([3,4]))
        self.assertTrue(subGraphs[4] == set([4]))
        self.assertTrue(subGraphs[5] == set([5]))
        self.assertTrue(subGraphs[6] == set([6,8]))
        self.assertTrue(subGraphs[7] == set([7]))
        self.assertTrue(subGraphs[8] == set([8]))
        self.assertTrue(subGraphs[9] == set([9, 7]))

if __name__ == '__main__':
    unittest.main()

