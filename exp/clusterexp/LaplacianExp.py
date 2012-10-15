#Look at some properties of the Laplacian matrix 

import sys 
import logging
import numpy
import itertools 
from apgl.graph import *
from exp.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from apgl.graph.GraphUtils import GraphUtils
from exp.clusterexp.BemolData import BemolData
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator

numpy.random.seed(21)
#numpy.seterr("raise")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=200, precision=3)

generator = ErdosRenyiGenerator(p=0.2)
numVertices = 20 
graph = SparseGraph(GeneralVertexList(numVertices))
graph = generator.generate(graph)

print(graph.getWeightMatrix())

components = graph.findConnectedComponents()
print(graph)
for i in range(len(components)): 
    print(components[i])

L = GraphUtils.shiftLaplacian(graph.getSparseWeightMatrix())

u, V = numpy.linalg.eig(L.todense())
inds = numpy.argsort(u)
u = u[inds]
print(u)
print(numpy.diff(u))