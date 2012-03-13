
"""
We want to find the most influencial vertices for the Campylobactor information diffusion
data. We will use different random graph models, and then output a ranked list of the
most influencial vertices in CSV format.

"""

import logging
import random
import sys
import time
import numpy
from apgl.io import *
from apgl.influence.GreedyInfluence import GreedyInfluence
from apgl.util import *
from apgl.graph import *
from apgl.generator import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

random.seed(21)
numpy.random.seed(21)
startTime = time.time()

numVertices = 200
#numVertices = 1000

ps = [0.2, float(30)/numVertices]
ks = [15]
generators = [SmallWorldGenerator(ps[0], ks[0]), ErdosRenyiGenerator(ps[1])]
numMethods = 3
numGraphTypes = len(graphTypes)

k = numVertices 

#Store a ranked list of vertex indices and cumulative influence 
influenceArray = numpy.zeros((k, 2))

outputDirectory = PathDefaults.getOutputDir()
outputDir = outputDirectory + "influence/"

for i in range(len(generators)):
    fileName = outputDir + "FullTransGraph" + str(generators[i]) + ".spg"
    graph = SparseGraph.load(fileName)
    numVertices = graph.getNumVertices()
    logging.info(graph.degreeDistribution())

    logging.info("Computing max product paths")
    #Make sure the diagonal entrices have information 1
    P = graph.maxProductPaths()
    P = P + numpy.eye(numVertices)

    logging.info("Computing max influence")
    influence = GreedyInfluence()
    inds = influence.maxInfluence(P, maxIks)

    for m in range(len(iks)):
        ik = iks[m]
        influenceArray[m, 0, i] = numpy.sum(numpy.max(P[inds[0:ik], :], 0))
        influenceArray[m, 1, i] = numpy.sum(numpy.max(P[inds2[0:ik], :], 0))
        influenceArray[m, 2, i] = numpy.sum(numpy.max(P[inds3[0:ik], :], 0))

#Now save the results
numpy.save(outputDir + "InfluenceArraySW1", influenceArray[:,:,0])
numpy.save(outputDir + "InfluenceArrayER", influenceArray[:,:,1])

logging.info("All done.")