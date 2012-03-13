'''
Created on 2 Jul 2009

@author: charanpal
'''

from apgl.util.Parameter import Parameter
from apgl.kernel.KernelUtils import KernelUtils
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
import numpy
import math 

class GeometricRandomGenerator(object):
    #Create with an empty graph
    def __init__(self, graph):
        self.graph = graph

    """
    A class which generates a random graph according to the following simple method.
    First, layout all vertices randomly according a uniform distribution on [0,1] in
    n dimensions. Then donote the probability of edges as an exponential function of
    the distance, i.e. p_ij = exp(-alpha d(v_i, v_j)). Then we add edges according to
    a constant probability (i.e. using an Erdos-Renyi process).
    """
    def generateGraph(self, alpha, p, dim):
        Parameter.checkFloat(alpha, 0.0, float('inf'))
        Parameter.checkFloat(p, 0.0, 1.0)
        Parameter.checkInt(dim, 0, float('inf'))
        
        numVertices = self.graph.getNumVertices()
        self.X = numpy.random.rand(numVertices, dim)

        D = KernelUtils.computeDistanceMatrix(numpy.dot(self.X, self.X.T))
        P = numpy.exp(-alpha * D)
        diagIndices = numpy.array(list(range(0, numVertices)))
        P[(diagIndices, diagIndices)] = numpy.zeros(numVertices)

        B = numpy.random.rand(numVertices, numVertices) <= P 

        #Note that B is symmetric - could just go through e.g. upper triangle 
        for i in range(numpy.nonzero(B)[0].shape[0]):
            v1 = numpy.nonzero(B)[0][i]
            v2 = numpy.nonzero(B)[1][i]
            
            self.graph.addEdge(v1, v2)

        erdosRenyiGenerator = ErdosRenyiGenerator(p)
        self.graph = erdosRenyiGenerator.generate(self.graph, False)

        return self.graph

    def generateGraph2(self, alpha, dim):
        numVertices = self.graph.getNumVertices()
        self.X = numpy.random.rand(numVertices, dim)

        D = KernelUtils.computeDistanceMatrix(numpy.dot(self.X, self.X.T))
        #P = beta**(-alpha * D)
        """
        Normalise the distance matrix so that the max distance (corner to corner)
        is 1. The value of k is the cumulative probability of an edge for any node
        """

        P = (D / math.sqrt(dim)) ** -alpha
        diagIndices = numpy.array(list(range(0, numVertices)))
        P[(diagIndices, diagIndices)] = numpy.zeros(numVertices)
        P = P / numpy.array([numpy.sum(P, 1)]).T 
        B = numpy.random.rand(numVertices, numVertices) <= P

        #Note that B is symmetric - could just go through e.g. upper triangle
        for i in range(numpy.nonzero(B)[0].shape[0]):
            v1 = numpy.nonzero(B)[0][i]
            v2 = numpy.nonzero(B)[1][i]

            self.graph.addEdge(v1, v2)

        return self.graph

    def getPositions(self):
        return self.X

    graph = None
    X = None

