"""
A script to create a learning dataset in which we can learn percolations and then
maximise influence on the resulting graph.

We have a graph of vertices of percolations with known decay. 
"""

from apgl.graph import *
from apgl.util import *
from apgl.data import * 
import numpy
import logging
import random
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

outputDir = PathDefaults.getOutputDir() + "influence/"
numVertices = 500
numFeatures = 5

random.seed(21)
numpy.random.seed(21)
noises = [0.0, 0.05, 0.1, 0.15, 0.2]

#Elements of X are in range [0, 1]
verticies = (numpy.random.rand(numVertices, numFeatures))
vList = VertexList(numVertices, numFeatures)
vList.setVertices(verticies)

p = 0.05
graph = SparseGraph(vList, False)
generator = ErdosRenyiGenerator(graph)
graph = generator.generateGraph(p)

#Now extract all pairs of vertices and assign values to edges
allEdges = graph.getAllEdges()
logging.info("Number of edges: " + str(allEdges.shape[0]))

egos = vList.getVertices(allEdges[:, 0])
alters = vList.getVertices(allEdges[:, 1])

C = numpy.random.rand(numFeatures, numFeatures)
C = C / numpy.sum(C)
cFilename = outputDir + "C.npy"
numpy.save(cFilename, C)
logging.info("Saved matrix of coefficients to " + cFilename)
alterCs = numpy.dot(egos, C)

#Make sure the real percolations are in the range [0,1]
yReal = numpy.sum(alters * alterCs, 1)
logging.info("min(y)= " + str(numpy.min(yReal)))
logging.info("max(y)= " + str(numpy.max(yReal)))

for noise in noises: 
    y = yReal + numpy.random.randn(allEdges.shape[0])*noise
    y[y>1] = 1
    y[y<0] = 0.01

    graph.addEdges(allEdges, y)

    #Now, save the results to a file, and also the graph
    outputFileName = outputDir + "SyntheticExamples_n=" + str(noise) + ".spg"
    graph.save(outputFileName)
