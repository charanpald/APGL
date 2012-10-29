"""
Use the clustering at the end of the graph using the exact clustering method
and then find modularities etc at the earlier clustering. 
"""

import sys
import logging
import numpy
from apgl.graph import *
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator

numpy.random.seed(21)
numpy.seterr("raise")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)

#Start off with the HIV data
hivReader = HIVGraphReader()
graph = hivReader.readHIVGraph()
fInds = hivReader.getIndicatorFeatureIndices()

#The set of edges indexed by zeros is the contact graph
#The ones indexed by 1 is the infection graph
edgeTypeIndex1 = 0
edgeTypeIndex2 = 1
sGraphContact = graph.getSparseGraph(edgeTypeIndex1)
sGraphInfect = graph.getSparseGraph(edgeTypeIndex2)
sGraphContact = sGraphContact.union(sGraphInfect)
graph = sGraphContact

#Find max component
components = graph.findConnectedComponents()
graph = graph.subgraph(list(components[0]))
#graph = graph.subgraph(range(500))
logging.debug(graph)

resultsDir = PathDefaults.getOutputDir() + "cluster/"

detectionIndex = fInds["detectDate"]
vertexArray = graph.getVertexList().getVertices()
detections = vertexArray[:, detectionIndex]

startYear = 1900
daysInMonth = 30
monthStep = 3
dayList = list(range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep))
dayList.append(numpy.max(detections))

subgraphIndicesList = []
subgraphIndicesList.append(range(graph.getNumVertices()))

k1 = 25
k2 = 2*k1
clusterer = IterativeSpectralClustering(k1, k2)
clusterer.nb_iter_kmeans = 20

logging.info("Running exact method")
iterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
clusterListExact, timeListExact, boundList = clusterer.clusterFromIterator(iterator, False, verbose=True)

clusters = clusterListExact[0]

subgraphIndicesList = []
#minGraphSize = 100
minGraphSize = 500

#Generate subgraph indices list
for i in dayList:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    if subgraphIndices.shape[0] >= minGraphSize:
        subgraphIndicesList.append(subgraphIndices)

numGraphs = len(subgraphIndicesList)
modularities = numpy.zeros(numGraphs)
kwayNormalisedCuts = numpy.zeros(numGraphs)

#Need to fix this to use the right 
fullW = graph.getWeightMatrix()
#i = 0
for i in range(len(subgraphIndicesList)):
    W = fullW[subgraphIndicesList[i], :][:, subgraphIndicesList[i]]

    modularities[i] = GraphUtils.modularity(W, clusters[subgraphIndicesList[i]])
    kwayNormalisedCuts[i] = GraphUtils.kwayNormalisedCut(W, clusters[subgraphIndicesList[i]])
    #i += 1

print(modularities)
print(kwayNormalisedCuts)

resultsFileName = resultsDir + "HIVResults2.npz"
file = open(resultsFileName, 'w')
numpy.savez(file, modularities, kwayNormalisedCuts)
logging.info("Saved file as " + resultsFileName)
