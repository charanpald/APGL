import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util import *
import networkx.algorithms.shortest_paths.dense as dense
import networkx.algorithms.shortest_paths.unweighted as unweighted

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class DenseGraphProfile(object):
    def __init__(self):
        numVertices = 1000
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = DenseGraph(vList)
        p = 0.01
        generator = ErdosRenyiGenerator(p)


        self.graph = generator.generate(graph)
        print(self.graph.getNumEdges())

    def profileNetworkXFloydWarshall(self):
        nxGraph = self.graph.toNetworkXGraph()

        ProfileUtils.profile('dense.floyd_warshall_numpy(nxGraph)', globals(), locals())

    def profileNetworkXShortestPaths(self):
        nxGraph = self.graph.toNetworkXGraph()

        ProfileUtils.profile('unweighted.all_pairs_shortest_path_length(nxGraph)', globals(), locals())

    def profileFloydWarshall(self):
        ProfileUtils.profile('self.graph.floydWarshall()', globals(), locals())

    def profileFindAllDistances(self):
        ProfileUtils.profile('self.graph.findAllDistances()', globals(), locals())

    def profileDijkstrasAlgorithm(self):
        n = 10
        A = self.graph.adjacencyMatrix()

        def runDijkstrasAlgorithm():
            for i in range(n):
                self.graph.dijkstrasAlgorithm(i, A)

        ProfileUtils.profile('runDijkstrasAlgorithm()', globals(), locals())

profiler = DenseGraphProfile()
#profiler.profileFindAllDistances() #28.3
profiler.profileFloydWarshall() #20.2
#profiler.profileNetworkXFloydWarshall() #11.04
#profiler.profileNetworkXShortestPaths()