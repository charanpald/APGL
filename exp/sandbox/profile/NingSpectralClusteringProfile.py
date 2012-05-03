
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.IterativeSpectralClustering import * 
from exp.sandbox.NingSpectralClustering import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class NingSpectralClusteringProfile(object):
    def __init__(self):
        numVertices = 250
        graph = SparseGraph(GeneralVertexList(numVertices))

        p = 0.1
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

        subgraphIndicesList = []
        for i in range(100, numVertices, 50):
            subgraphIndicesList.append(range(i))

        k = 5

        self.graph = graph
        self.subgraphIndicesList = subgraphIndicesList
        self.clusterer = NingSpectralClustering(k)

        #Try a sequence of graphs which don't change much


    def profileCluster(self):
        iterator = IncreasingSubgraphListIterator(self.graph, self.subgraphIndicesList)

        ProfileUtils.profile('self.clusterer.cluster(iterator)', globals(), locals())

    def profileCluster2(self):
        numVertices = 250
        graph = SparseGraph(GeneralVertexList(numVertices))

        p = 0.1
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)
        
        W = graph.getWeightMatrix()

        WList = []

        for i in range(50):
            s = numpy.random.randint(0, numVertices)
            t = numpy.random.randint(0, numVertices)
            logging.info(s, t)
            W[s, t] += 0.5
            W[t, s] += 0.5 
            WList.append(W.copy())

        iterator = iter(WList)

        ProfileUtils.profile('self.clusterer.cluster(iterator)', globals(), locals())


profiler = NingSpectralClusteringProfile()
profiler.profileCluster() #12
profiler.profileCluster2() 
