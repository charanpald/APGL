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
        numVertices = 5000

        self.graph = PySparseGraph(numVertices)
        
        numEdges = 100000
        edges = numpy.zeros((numEdges, 2))
        edges[:, 0] = numpy.random.randint(0, numVertices, numEdges)
        edges[:, 1] = numpy.random.randint(0, numVertices, numEdges)
        
        self.graph.addEdges(edges)
        
        

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
        
    def profileConnectedComponents(self):
        ProfileUtils.profile('self.graph.findConnectedComponents()', globals(), locals())

profiler = PySparseGraphProfile()
profiler.profileConnectedComponents()
#Takes 24.8