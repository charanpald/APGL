
"""
Compare the perturbation bound on the HIV data to the exact error. 
"""
import sys 
import logging
import numpy
from apgl.graph import *
from exp.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering

numpy.random.seed(21)
#numpy.seterr("raise")
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

resultsDir = PathDefaults.getOutputDir() + "cluster/"

detectionIndex = fInds["detectDate"]
vertexArray = graph.getVertexList().getVertices()
detections = vertexArray[:, detectionIndex]

startYear = 1900
daysInMonth = 30
monthStep = 1
dayList = list(range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep))
dayList.append(numpy.max(detections))

subgraphIndicesList = []
minGraphSize = 150
maxGraphSize = 500 

#Generate subgraph indices list 
for i in dayList:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    if subgraphIndices.shape[0] >= minGraphSize and subgraphIndices.shape[0] <= maxGraphSize: 
        subgraphIndicesList.append(subgraphIndices)


def getIterator():
    return IncreasingSubgraphListIterator(graph, subgraphIndicesList)
    
datasetName = "HIV"
numGraphs = len(subgraphIndicesList)

k1 = 25 
k2 = 2*k1

clusterer = IterativeSpectralClustering(k1, k2)
clusterer.nb_iter_kmeans = 20
clusterer.computeBound = True 
iterator = getIterator() 
clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)

boundList = numpy.array(boundList)
print(boundList)
