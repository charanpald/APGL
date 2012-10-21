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
from apgl.util.PathDefaults import PathDefaults
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from apgl.graph.GraphUtils import GraphUtils
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.util.Util import Util 

numpy.random.seed(21)
#numpy.seterr("raise")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60, precision=3)

numVertices = 100 
graph = SparseGraph(GeneralVertexList(numVertices))

p = 0.2 
k = 10 
generator = SmallWorldGenerator(p, k)
graph = generator.generate(graph)
graph2 = graph.copy()

changeEdges = 5 
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

u = numpy.flipud(numpy.sort(u))
u2 = numpy.flipud(numpy.sort(u2))
u3 = numpy.flipud(numpy.sort(u3))

Lroot = Util.matrixPower(L.todense(), -0.5)
bound = numpy.linalg.norm(Lroot.dot(B).dot(Lroot), 2)
print(bound)

print(numpy.trace(L.todense()))

print(u)
print(numpy.abs(numpy.diff(u)))
print(u2)
print(u3)


print(graph)
print(graph2)

k1 = numpy.argmax(numpy.abs(numpy.diff(u)))+1
k2 = k1

logging.debug("k=" + str(k1))
iterator = iter([W, W2])

clusterer = IterativeSpectralClustering(k1, k2)
clusterer.nb_iter_kmeans = 20
clusterer.computeBound = True 
logging.debug("Starting clustering")
clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)

boundList = numpy.array(boundList)
print(boundList)

#TODO: Use 3 Erdos Renyi graphs and concantenate, then add random edges 
#Create class for this

