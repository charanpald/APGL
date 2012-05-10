
"""
Compare our clustering method and that of Ning et al. on the HIV data.
"""
import sys 
import logging
import numpy
from apgl.graph import *
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator

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
monthStep = 3
dayList = list(range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep))
dayList.append(numpy.max(detections))

subgraphIndicesList = []
minGraphSize = 500

#Generate subgraph indices list 
for i in dayList:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    if subgraphIndices.shape[0] >= minGraphSize: 
        subgraphIndicesList.append(subgraphIndices)

#subgraphIndicesList2 = []
#for i in range(5):
#    subgraphIndicesList2.append(subgraphIndicesList[i])
#subgraphIndicesList = subgraphIndicesList2

def getIterator():
    return IncreasingSubgraphListIterator(graph, subgraphIndicesList)
    
datasetName = "HIV"
numGraphs = len(subgraphIndicesList)

clusterExpHelper = ClusterExpHelper(getIterator, datasetName, numGraphs)
clusterExpHelper.runIASC = False
clusterExpHelper.runExact = False
clusterExpHelper.runModularity = False
clusterExpHelper.runNystrom = True
clusterExpHelper.runNing = False
clusterExpHelper.k1 = 25
clusterExpHelper.k2 = 2*clusterExpHelper.k1

clusterExpHelper.runExperiment()
