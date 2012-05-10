import numpy
import logging
import sys
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from apgl.util.PathDefaults import PathDefaults
from exp.clusterexp.BemolData import BemolData

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class IterativeSpectralClusteringProfile(object):
    def __init__(self):
        numVertices = 1000
        graph = SparseGraph(GeneralVertexList(numVertices))

        p = 0.1
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

        subgraphIndicesList = []
        for i in range(100, numVertices, 10):
            subgraphIndicesList.append(range(i))

        k1 = 5
        k2 = 20 

        self.graph = graph
        self.subgraphIndicesList = subgraphIndicesList
        self.clusterer = IterativeSpectralClustering(k1, k2)


    def profileClusterFromIterator(self):
        iterator = IncreasingSubgraphListIterator(self.graph, self.subgraphIndicesList)
        dataDir = PathDefaults.getDataDir() + "cluster/"
        #iterator = getBemolGraphIterator(dataDir)

        ProfileUtils.profile('self.clusterer.clusterFromIterator(iterator)', globals(), locals())

profiler = IterativeSpectralClusteringProfile()
profiler.profileClusterFromIterator() #19.7 


# python -c "execfile('exp/sandbox/profile/IterativeSpectralClusteringProfile.py')"
