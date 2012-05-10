
from exp.sandbox.IterativeModularityClustering import IterativeModularityClustering
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator
from apgl.graph import *
from apgl.generator import *
import unittest
import numpy
import logging
import sys
import itertools


class  IterativeModularityClusteringTestCase(unittest.TestCase):
    def testClusterFromIterator(self):
        #Create a small graph and try the iterator increasing the number of vertices.
        numVertices = 50
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        ell = 2
        m = 2
        generator = BarabasiAlbertGenerator(ell, m)
        graph = generator.generate(graph)

        indices = numpy.random.permutation(numVertices)
        subgraphIndicesList = [indices[0:5], indices]

        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)

        #Try a much longer sequence of vertices
        subgraphIndicesList = []
        for i in range(10, numVertices):
            subgraphIndicesList.append(range(i))

        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)

        k = 4
        clusterer = IterativeModularityClustering(k)
        clustersList = clusterer.clusterFromIterator(graphIterator)

        logging.info(clustersList)

if __name__ == '__main__':
    unittest.main()

