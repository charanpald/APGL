"""
Observe the effect in the perturbations of Laplacians 
"""

import sys 
import logging
import numpy
import scipy 
import itertools 
import copy
from apgl.graph import *
from exp.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from apgl.graph.GraphUtils import GraphUtils
from exp.clusterexp.BemolData import BemolData
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.util.Util import Util 

numpy.random.seed(21)
#numpy.seterr("raise")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)

numVertices = 100 
graph = SparseGraph(GeneralVertexList(numVertices))

p = 0.1 
k = 5 
generator = SmallWorldGenerator(p, k)
graph = generator.generate(graph)

graph2 = graph.copy()


changeEdges = 10 
i = 0

while i < changeEdges: 
    inds = numpy.random.randint(0, numVertices, 2)
    if graph2[inds[0], inds[1]] == 0: 
        graph2[inds[0], inds[1]] = 1
        i += 1 

W = graph.getSparseWeightMatrix()
W2 = graph2.getSparseWeightMatrix()

L = GraphUtils.shiftLaplacian(W)
L2 = GraphUtils.shiftLaplacian(W2)

u, V = numpy.linalg.eig(L.todense())
u2, V2 = numpy.linalg.eig(L2.todense())

B = (L2-L).todense()
u3, V3 = numpy.linalg.eig(B)
#print(B)


u = numpy.sort(u)
u2 = numpy.sort(u2)
u3 = numpy.sort(u3)

Lroot = Util.matrixPower(L.todense(), -0.5)
bound = numpy.linalg.norm(Lroot.dot(B).dot(Lroot), 2)
print(bound)

print(numpy.trace(L.todense()))

print(u)
print(u2)
print(u3)

print(numpy.diff(u))
print(graph)
print(graph2)


