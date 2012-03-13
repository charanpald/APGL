"""
Here we generate a number of graphs according to the configuration model and fit
the degree sequence to the HIV contact graph.
"""
import logging
import sys
import numpy
from apgl.graph import *
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.DateUtils import DateUtils
from apgl.util.PathDefaults import PathDefaults 
from apgl.generator.ConfigModelGenerator import ConfigModelGenerator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=150)
numpy.random.seed(21)

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
sGraph = sGraphContact

startYear = 1900
daysInYear = 365
daysInMonth = 30 
monthStep = 3

detectionIndex = fInds["detectDate"]
vertexArray = sGraph.getVertexList().getVertices()
detections = vertexArray[:, detectionIndex]

#Make sure we include all detections
dayList = range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep)
dayList.append(numpy.max(detections))

#The config graph has the same number of edges and vertices at each time point
numGraphs = 10
numVertices = sGraph.getNumVertices()

resultsDir = PathDefaults.getOutputDir() + "viroscopy/"

def computeContactConfigGraphs():
    graphFileNameBase = resultsDir + "ConfigGraph"

    for j in range(numGraphs):
        configGraph = SparseGraph(GeneralVertexList(numVertices))
        degSequence = numpy.zeros(numVertices, numpy.int)
        lastDegSequence = numpy.zeros(numVertices, numpy.int)
        generator = ConfigModelGenerator(lastDegSequence)

        for i in dayList:
            logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
            subgraphIndices = numpy.nonzero(detections <= i)[0]
            subgraphIndices = numpy.unique(subgraphIndices)
            subgraph = sGraph.subgraph(subgraphIndices)

            subDegSequence = subgraph.degreeSequence()
            degSequence[subgraphIndices] = subDegSequence
            diffSequence = degSequence - lastDegSequence
            generator.setOutDegSequence(diffSequence)
            configGraph = generator.generate(configGraph, False)

            lastDegSequence = configGraph.degreeSequence()
            assert (degSequence>=lastDegSequence).all()
            assert subgraph.getNumEdges() >= configGraph.getNumEdges()

        configGraph.save(graphFileNameBase + str(j))

numpy.random.seed(21)

def computeInfectConfigGraphs():
    #We need the directed infection graph 
    hivReader = HIVGraphReader()
    graph = hivReader.readHIVGraph(False)
    sGraphInfect = graph.getSparseGraph(edgeTypeIndex2)
    sGraph = sGraphInfect

    graphFileNameBase = resultsDir + "ConfigInfectGraph"

    for j in range(numGraphs):
        configGraph = SparseGraph(GeneralVertexList(numVertices), False)
        
        outDegSequence = numpy.zeros(numVertices, numpy.int)
        inDegSequence = numpy.zeros(numVertices, numpy.int)
        lastOutDegSequence = numpy.zeros(numVertices, numpy.int)
        lastInDegSequence = numpy.zeros(numVertices, numpy.int)
        generator = ConfigModelGenerator(lastOutDegSequence, lastInDegSequence)

        for i in dayList:
            logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
            subgraphIndices = numpy.nonzero(detections <= i)[0]
            subgraphIndices = numpy.unique(subgraphIndices)
            subgraph = sGraph.subgraph(subgraphIndices)

            outDegSequence[subgraphIndices] = subgraph.outDegreeSequence()
            inDegSequence[subgraphIndices] = subgraph.inDegreeSequence()
            outDiffSequence = outDegSequence - lastOutDegSequence
            inDiffSequence = inDegSequence - lastInDegSequence

            generator.setInDegSequence(inDiffSequence)
            generator.setOutDegSequence(outDiffSequence)
            configGraph = generator.generate(configGraph, False)

            lastOutDegSequence = configGraph.outDegreeSequence()
            lastInDegSequence = configGraph.inDegreeSequence()

            assert (outDegSequence>=lastOutDegSequence).all()
            assert (inDegSequence>=lastInDegSequence).all()

        configGraph.save(graphFileNameBase + str(j))

computeContactConfigGraphs()
#computeInfectConfigGraphs()