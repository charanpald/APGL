import numpy
import logging
import sys
import scipy.sparse
from apgl.graph import *
from apgl.generator import *
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class PySparseGraphProfile(object):
    def __init__(self):
        numVertices = 1000
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = PySparseGraph(vList)
        p = 0.4
        generator = ErdosRenyiGenerator(p)

        self.graph = generator.generate(graph)

    def profileDijkstrasAlgorithm(self):
        n = 10

        def runDijkstrasAlgorithm():
            for i in range(n):
                self.graph.dijkstrasAlgorithm(i)

        ProfileUtils.profile('runDijkstrasAlgorithm()', globals(), locals())

    def profileFloydWarshall(self):
        ProfileUtils.profile('self.graph.floydWarshall()', globals(), locals())

    def profileGetWeightMatrix(self):
        ProfileUtils.profile('self.graph.getWeightMatrix()', globals(), locals())

profiler = PySparseGraphProfile()
profiler.profileDijkstrasAlgorithm()
#Takes 24.8