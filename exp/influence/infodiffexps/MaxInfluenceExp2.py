
"""
We compare the influence methods in a dense transmission graph. 
"""

import logging
import random
import sys
import time
import numpy

from apgl.graph import * 
from apgl.influence.GreedyInfluence import GreedyInfluence
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

random.seed(21)
numpy.random.seed(21)
startTime = time.time()

numVertices = 1000
numMethods = 3


#iks = range(10, 100, 10)
iks = list(range(10, 510, 10))
iks.insert(0, 1)
maxIks = max(iks)

influenceArray = numpy.zeros((len(iks), numMethods))

outputDirectory = PathDefaults.getOutputDir()
outputDir = outputDirectory + "influence/"

graphType="SmallWorld"
k = 50
p = 0.2

fileName = outputDir + "FullTransGraph" + graphType + "_" + str(p) + "_" + str(k) + ".spg"
graph = SparseGraph.load(fileName)
numVertices = graph.getNumVertices()

#Rewrite edges to have a decay rate of d
edges = graph.getAllEdges()
for m in range(edges.shape[0]):
    d = numpy.random.rand()
    graph.addEdge(edges[m, 0], edges[m, 1], d)

logging.info("Computing max product paths")
#Make sure the diagonal entrices have information 1
P = graph.maxProductPaths()
P = P + numpy.eye(numVertices)
P2 = numpy.array(P != 0, numpy.int32)

logging.info("Computing max influence")
influence = GreedyInfluence()
inds = influence.maxInfluence(P, maxIks)
logging.info("Computing Kempes max influence")
inds2 = influence.maxInfluence(P2, maxIks)
inds3 = numpy.random.permutation(numVertices)[0:maxIks]

for m in range(len(iks)):
    ik = iks[m]
    influenceArray[m, 0] = numpy.sum(numpy.max(P[inds[0:ik], :], 0))
    influenceArray[m, 1] = numpy.sum(numpy.max(P[inds2[0:ik], :], 0))
    influenceArray[m, 2] = numpy.sum(numpy.max(P[inds3[0:ik], :], 0))

#Now save the results
numpy.save(outputDir + "influenceArraySW2", influenceArray)

logging.info("All done.")