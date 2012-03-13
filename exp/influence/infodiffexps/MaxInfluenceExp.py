
"""
We create a small world graph, then run the simulation for 10 iterations. Then we
maximise the quality of information.
Compare against Kempe method and random. Can compare against exaustive solution?
"""

import logging
import random
import sys
import time

from apgl.io import *
from apgl.influence.GreedyInfluence import GreedyInfluence
from apgl.util import *
from apgl.graph import *
from apgl.generator import * 
 
import numpy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

random.seed(21)
numpy.random.seed(21)
startTime = time.time()

#We will just read the graph files and then perform maximum influence
#Can we learn which are the most influencial nodes and improve on greedy method?
#Note we need the SVM to compute transitions for all edges!

#numVertices = 200
numVertices = 1000

graphTypes = ["SmallWorld", "ErdosRenyi"]
ps = [0.2, float(30)/numVertices]
ks = [15]
generators = [SmallWorldGenerator(ps[0], ks[0]), ErdosRenyiGenerator(ps[1])]
numMethods = 3
numGraphTypes = len(graphTypes)

#iks = range(10, 100, 10)
iks = list(range(10, 510, 10))
iks.insert(0, 1)
maxIks = max(iks)

influenceArray = numpy.zeros((len(iks), numMethods, numGraphTypes))

outputDirectory = PathDefaults.getOutputDir()
outputDir = outputDirectory + "influence/"

for i in range(len(generators)):    
    vList = VertexList(numVertices, 0)

    fileName = outputDir + "FullTransGraph" + str(generators[i]) + ".spg"
    graph = SparseGraph.load(fileName)
    numVertices = graph.getNumVertices()
    logging.info(graph.degreeDistribution())

    writer = PajekWriter()
    writer.writeToFile(outputDir + "FullTransGraph" +  str(generators[i]), graph)

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
        influenceArray[m, 0, i] = numpy.sum(numpy.max(P[inds[0:ik], :], 0))
        influenceArray[m, 1, i] = numpy.sum(numpy.max(P[inds2[0:ik], :], 0))
        influenceArray[m, 2, i] = numpy.sum(numpy.max(P[inds3[0:ik], :], 0))
    
#Now save the results
numpy.save(outputDir + "InfluenceArraySW1", influenceArray[:,:,0])
numpy.save(outputDir + "InfluenceArrayER", influenceArray[:,:,1])

logging.info("All done.")