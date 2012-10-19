#Look at some properties of the Laplacian matrix 

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

numpy.random.seed(21)
#numpy.seterr("raise")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=300, precision=3)

numClusters = 3
startClusterSize = 20
endClusterSize = 60
clusterStep = 5
#clusterStep = 20

numVertices = numClusters*endClusterSize
vList = GeneralVertexList(numVertices)

subgraphIndicesList = [range(0, startClusterSize)]
subgraphIndicesList[0].extend(range(endClusterSize, endClusterSize+startClusterSize))
subgraphIndicesList[0].extend(range(2*endClusterSize, 2*endClusterSize+startClusterSize))
for i in range(startClusterSize, endClusterSize-clusterStep+1, clusterStep):
	subgraphIndices = copy.copy(subgraphIndicesList[-1])
	subgraphIndices.extend(range(i, i+clusterStep))
	subgraphIndices.extend(range(endClusterSize+i, endClusterSize+i+clusterStep))
	subgraphIndices.extend(range(2*endClusterSize+i, 2*endClusterSize+i+clusterStep))
	subgraphIndicesList.append(subgraphIndices)


tmp = copy.copy(subgraphIndicesList[:-1])
tmp.reverse()
subgraphIndicesList.extend(tmp)

p = 0.05
pClust = 0.3

W = numpy.ones((numVertices, numVertices))*p
for i in range(numClusters):
	W[endClusterSize*i:endClusterSize*(i+1), endClusterSize*i:endClusterSize*(i+1)] = pClust
P = numpy.random.rand(numVertices, numVertices)
W = numpy.array(P < W, numpy.float)
upTriInds = numpy.triu_indices(numVertices)
W[upTriInds] = 0
W = W + W.T
graph = SparseGraph(vList)
graph.setWeightMatrix(W)

L = GraphUtils.shiftLaplacian(scipy.sparse.csr_matrix(W))
u, V = numpy.linalg.eig(L.todense())
print(V.shape)
print(numpy.linalg.cond(V))

# run with exact eigenvalue decomposition
logging.info("Running exact method")
graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)

"""
for W in graphIterator: 
    graph = SparseGraph(GeneralVertexList(W.shape[0]))
    graph.setWeightMatrixSparse(W) 
    components = graph.findConnectedComponents()
    print(graph)
    
    
    L = GraphUtils.shiftLaplacian(graph.getSparseWeightMatrix())
    
    u, V = numpy.linalg.eig(L.todense())
    inds = numpy.argsort(u)
    u = u[inds]
    
    k = 20 
    print((u[0:k]**2).sum())
    print((u[k:]**2).sum())
"""

numGraphs = len(subgraphIndicesList)

k1 = 3 
k2 = 3

clusterer = IterativeSpectralClustering(k1, k2)
clusterer.nb_iter_kmeans = 20
clusterer.computeBound = True 
clusterList, timeList, boundList = clusterer.clusterFromIterator(graphIterator, verbose=True)

boundList = numpy.array(boundList)
print(boundList)