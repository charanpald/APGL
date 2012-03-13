
"""
Cluster the HIV contact graph according to its growth
"""

import sys
import logging
import numpy
from apgl.graph import *
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from apgl.clustering.SpectralClusterer import SpectralClusterer

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
logging.debug(graph)

resultsDir = PathDefaults.getOutputDir() + "viroscopy/"

detectionIndex = fInds["detectDate"]
vertexArray = graph.getVertexList().getVertices()
detections = vertexArray[:, detectionIndex]

startYear = 1900
daysInMonth = 30
monthStep = 3
dayList = list(range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep))
dayList.append(numpy.max(detections))

subgraphIndicesList = []
minGraphSize = 500

#Generate subgraph indices list
for i in dayList:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    subgraphIndicesList.append(subgraphIndices)

saveResults = False
clusteringsFileName = resultsDir + "HIVClustering.npy"

if saveResults:
    #Should optimise over k by using k-way normalised cut 
    k = 30
    clusterer = SpectralClusterer(k)
    clusterings = clusterer.cluster(graph)

    file = open(clusteringsFileName, 'w')
    numpy.save(file, clusterings)
    file.close()
else:
    file = open(clusteringsFileName, 'r')
    clusterings = numpy.load(file)
    file.close()


clusterIds = numpy.unique(clusterings)
print("Number of clusters: " + str(clusterIds.shape[0]))
clusterSizes = numpy.zeros(clusterIds.shape[0])

for i in range(clusterIds.shape[0]):
    clusterSizes[i] = numpy.nonzero(clusterings==clusterIds[i])[0].shape[0]

print(clusterSizes)
print(numpy.sort(clusterSizes))

#See if any of the clusters are "abnormal" in terms of MSM population, and detection

