import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class MatrixGraphProfile(object):
    def __init__(self):
        self.n = 1000
        self.m = 10000
        numFeatures = 0
        vList = VertexList(self.n, numFeatures)
        #self.graph = SparseGraph(vList)
        #self.graph = DenseGraph(vList)
        self.graph = PySparseGraph(vList)


    def profileAddEdge(self):
        V = numpy.random.randint(0, self.n, (self.m,2))
        u = numpy.random.rand(self.m)

        def runAdd():
            for i in range(self.m):
                self.graph.addEdge(V[i,0], V[i,1], u[i])

        ProfileUtils.profile('runAdd()', globals(), locals())

    def profileGetEdge(self):
        V = numpy.random.randint(0, self.n, (self.m,2))
        u = numpy.random.rand(self.m)

        numEdges = 1000
        for i in range(numEdges):
            self.graph.addEdge(V[i,0], V[i,1], u[i])


        def runGet():
            for i in range(self.m):
                u = self.graph.getEdge(V[i,0], V[i,1])

        ProfileUtils.profile('runGet()', globals(), locals())

    def profileNeighbours(self):
        V = numpy.random.randint(0, self.n, (self.m,2))
        u = numpy.random.rand(self.m)

        numEdges = 1000
        for i in range(numEdges):
            self.graph.addEdge(V[i,0], V[i,1], u[i])

        v  = numpy.random.randint(0, self.n, self.m)

        def runNeighbours():
            for i in range(self.m):
                u = self.graph.neighbours(v[i])

        ProfileUtils.profile('runNeighbours()', globals(), locals())

profiler = MatrixGraphProfile()
#profiler.profileGetEdge()
profiler.profileNeighbours()

#PySparseGraph much faster 