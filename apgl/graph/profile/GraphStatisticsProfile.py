
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class GraphStatisticsProfile:
    def __init__(self):
        numVertices = 5000
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)
        
        p = 0.01
        generator = ErdosRenyiGenerator(p)
        self.graph = generator.generate(graph)

        self.statistics = GraphStatistics()

    def profileScalarStatistics(self):
        ProfileUtils.profile('self.statistics.scalarStatistics(self.graph, slowStats=False)', globals(), locals())

    def profileVectorStatistics(self):
        ProfileUtils.profile('self.statistics.vectorStatistics(self.graph)', globals(), locals())

profiler = GraphStatisticsProfile()
profiler.profileScalarStatistics()
#profiler.profileVectorStatistics()