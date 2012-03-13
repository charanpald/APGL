
import logging
import unittest
import numpy 
import apgl
import sys
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph

from apgl.clustering.SpectralClusterer import SpectralClusterer

class  SpectralClustererTestCase(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=False, precision=3)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testUnNormSpectralClusterer(self):
        numVertices = 10
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        #We form two cliques with an edge between then
        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(0, 3)
        graph.addEdge(1, 2)
        graph.addEdge(1, 3)
        graph.addEdge(2, 3)

        graph.addEdge(2, 4)

        graph.addEdge(4, 5)
        graph.addEdge(4, 6)
        graph.addEdge(4, 7)
        graph.addEdge(5, 6)
        graph.addEdge(5, 7)
        graph.addEdge(6, 7)
        graph.addEdge(7, 8)
        graph.addEdge(7, 9)

        #graph.addEdge(0, 4)

        k = 3
        clusterer = SpectralClusterer(k)
        clusters = clusterer.cluster(graph)

        self.assertEquals(clusters.shape[0], numVertices)
        self.assertEquals(numpy.unique(clusters).shape[0], k)
        logging.debug(clusters)

        realClusters = numpy.array([1,1,1,1, 0,0,0,0, 2,2])

        similarityMatrix1 = numpy.zeros((numVertices, numVertices))
        similarityMatrix2 = numpy.zeros((numVertices, numVertices))

        for i in range(numVertices):
            for j in range(numVertices):
                if clusters[i] == clusters[j]:
                    similarityMatrix1[i, j] = 1

                if realClusters[i] == realClusters[j]:
                    similarityMatrix2[i, j] = 1     

        self.assertTrue((similarityMatrix1 == similarityMatrix2).all())

 
if __name__ == '__main__':
    unittest.main()

