import logging
import random
import sys
import time
import numpy

from apgl.io import *
from apgl.influence.GreedyInfluence import GreedyInfluence
from apgl.influence.BABInfluence import BABInfluence
from apgl.util import *

sys.setrecursionlimit(1000)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

random.seed(21)
numpy.random.seed(21)
startTime = time.time()

outputDirectory = PathDefaults.getOutputDir()
outputDir = outputDirectory + "influence/"
fileName = outputDir + "FullTransGraphSmallWorld_0.2_15.txt"

reader = SimpleGraphReader()
graph = reader.readFromFile(fileName)
numVertices = graph.getNumVertices()
logging.info(graph.degreeDistribution())

#Take a subgraph
subgraphIndices = numpy.random.permutation(numVertices)
numVertices = 100
subgraphIndices = subgraphIndices[0:100]
graph = graph.subgraph(subgraphIndices)

d = 0.5

#Rewrite edges to have a decay rate of d
edges = graph.getAllEdges()
for m in range(edges.shape[0]):
    graph.addEdge(edges[m, 0], edges[m, 1], d)

logging.info("Computing max product paths")
#Make sure the diagonal entrices have information 1
P = graph.maxProductPaths()
P = P + numpy.eye(numVertices)

u = numpy.random.rand(numVertices)
L = 0.1*numpy.sum(u)

logging.info("Computing max budgeted influence with branch and bound method")
influence2 = BABInfluence()
inds2 = influence2.maxBudgetedInfluence(P, u, L)
logging.info("Best activation: " + str(numpy.sum(numpy.max(P[inds2, :], 0))))
logging.info("Cost: " + str(numpy.sum(u[inds2, :], 0)) + "\n")

logging.info("Computing max budgeted influence with greedy method")
influence = GreedyInfluence()
inds = influence.maxBudgetedInfluence(P, u, L)
logging.info("Best activation: " + str(numpy.sum(numpy.max(P[inds, :], 0))))
logging.info("Cost: " + str(numpy.sum(u[inds, :], 0)) + "\n\n")

#This is really slow even with 100 vertices - need a better method to cull 