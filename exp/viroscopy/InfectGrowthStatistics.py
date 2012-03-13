import logging
import sys
import matplotlib
import numpy
import os
import gc 
from datetime import date
#matplotlib.use('Qt4Agg') # do this before importing pylab
import matplotlib.pyplot as plt
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from apgl.util.Latex import Latex
from apgl.util.Util import Util
from apgl.graph import *
from apgl.viroscopy.HIVGraphReader import HIVGraphReader

"""
This script computes some basic statistics on the growing infection graph.
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=100, precision=3)

undirected = False 
hivReader = HIVGraphReader()
graph = hivReader.readHIVGraph(undirected, indicators=False)
fInds = hivReader.getNonIndicatorFeatureIndices()


figureDir = PathDefaults.getOutputDir() + "viroscopy/figures/infect/"
resultsDir = PathDefaults.getOutputDir() + "viroscopy/"

#The set of edges indexed by zeros is the contact graph
#The ones indexed by 1 is the infection graph
edgeTypeIndex1 = 0
edgeTypeIndex2 = 1
sGraphContact = graph.getSparseGraph(edgeTypeIndex1)
sGraphInfect = graph.getSparseGraph(edgeTypeIndex2)

sGraph = sGraphInfect
#sGraph = sGraph.subgraph(range(0, 500))

graphStats = GraphStatistics()
statsArray = graphStats.scalarStatistics(sGraph, False)
slowStats = True
saveResults = False

logging.info(sGraph)
logging.info("Number of vertices: " + str(statsArray[graphStats.numVerticesIndex]))
logging.info("Number of features: " + str(sGraph.getVertexList().getNumFeatures()))
logging.info("Number of edges: " + str(statsArray[graphStats.numEdgesIndex]))
logging.info("Largest component is " + str(statsArray[graphStats.maxComponentSizeIndex]))
logging.info("Number of components " + str(statsArray[graphStats.numComponentsIndex]))

#sGraph = sGraph.subgraph(components[componentIndex])
vertexArray = sGraph.getVertexList().getVertices(list(range(0, sGraph.getVertexList().getNumVertices())))
logging.info("Size of graph we will use: " + str(sGraph.getNumVertices()))

#Some indices
dobIndex = fInds["birthDate"]
detectionIndex = fInds["detectDate"]
deathIndex = fInds["deathDate"]
genderIndex = fInds["gender"]
orientationIndex = fInds["orient"]
locationIndex = fInds["province"]

detections = vertexArray[:, detectionIndex]

startYear = 1900
daysInYear = 365
daysInMonth = 30

q = 0.9

#This is a set of days to record simple statistics 
monthStep = 3
dayList = list(range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep))
dayList.append(numpy.max(detections))

absDayList = [float(i-numpy.min(detections)) for i in dayList]
subgraphIndicesList = []

#Locations and labels for years
locs = list(range(0, int(absDayList[-1]), daysInYear*2))
labels = numpy.arange(1986, 2006, 2)

#This is a set of days to record more complex vectorial statistics
monthStep2 = 60
dayList2 = [DateUtils.getDayDelta(date(1989, 12, 31), startYear)]
dayList2.append(DateUtils.getDayDelta(date(1993, 12, 31), startYear))
dayList2.append(DateUtils.getDayDelta(date(1997, 12, 31), startYear))
dayList2.append(DateUtils.getDayDelta(date(2001, 12, 31), startYear))
dayList2.append(int(numpy.max(detections)))
subgraphIndicesList2 = []

logging.info(dayList2)

for i in dayList2:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    subgraphIndicesList2.append(subgraphIndices)

logging.info(dayList)
plotInd = 1
plotStyles = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-']
plotStyles2 = ['k-', 'r-', 'g-', 'b-', 'c-', 'm-']
plotStyles3 = ['k-', 'k--', 'k-.', 'k:']
plotStyles4 = ['r-', 'r--', 'r-.', 'r:']
plotStyles5 = ['r-', 'b-', 'g-', 'r:']

for i in dayList:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    subgraphIndicesList.append(subgraphIndices)

#Check some properties of the trees 
def plotTreeStats():
    logging.info("Computing tree stats")
    resultsFileName = resultsDir + "InfectGrowthTreeStats.pkl"

    if saveResults:
        statsDictList = []

        for j in range(len(subgraphIndicesList2)):
            Util.printIteration(j, 1, len(subgraphIndicesList2))
            subgraphIndices = subgraphIndicesList2[j]
            subgraph = sGraph.subgraph(subgraphIndices)
            logging.info("Finding trees")
            trees = subgraph.findTrees()
            logging.info("Computing tree statistics")
            statsDict = {}

            locationEntropy = []
            orientEntropy = []
            detectionRanges = []

            for i in range(len(trees)):
                if len(trees[i]) > 1:
                    treeGraph = subgraph.subgraph(trees[i])
                    vertexArray = treeGraph.getVertexList().getVertices(list(range(treeGraph.getNumVertices())))
                    
                    locationEntropy.append(Util.entropy(vertexArray[:, locationIndex]))
                    orientEntropy.append(Util.entropy(vertexArray[:, orientationIndex]))
                    
                    detections = vertexArray[:, detectionIndex]
                    detectionRanges.append(numpy.max(detections) - numpy.min(detections))

            statsDict["locationEnt"] = numpy.array(locationEntropy)
            statsDict["orientEnt"] = numpy.array(orientEntropy)
            statsDict["detectRanges"] = numpy.array(detectionRanges)
            statsDictList.append(statsDict)

        Util.savePickle(statsDictList, resultsFileName, True)
    else:
        statsDictList = Util.loadPickle(resultsFileName)
        
        locBins = numpy.arange(0, 2.4, 0.2)
        detectBins = numpy.arange(0, 6500, 500)
        locationEntDists = []
        orientEntDists = []
        detectionDists = [] 

        for j in range(0, len(dayList2)):
            dateStr = (str(DateUtils.getDateStrFromDay(dayList2[j], startYear)))
            logging.info(dateStr)
            statsDict = statsDictList[j]
            plotInd2 = plotInd

            locationEntDists.append(statsDict["locationEnt"])
            orientEntDists.append(statsDict["orientEnt"])
            detectionDists.append(statsDict["detectRanges"])

        #for j in range(len(orientEntDists)):
        #    print(numpy.sum(numpy.histogram(orientEntDists[j])[0]))
        #    print(numpy.histogram(orientEntDists[j])[0]/float(orientEntDists[j].shape[0]))

        dateStrs = [DateUtils.getDateStrFromDay(dayList2[i], startYear) for i in range(1, len(dayList2))]

        plt.figure(plotInd2)
        histOut = plt.hist(locationEntDists, locBins, normed=True)
        plt.xlabel("Location Entropy")
        plt.ylabel("Probability Density")
        plt.savefig(figureDir + "LocationEnt" +  ".eps")
        #plt.legend()
        plotInd2 += 1

        plt.figure(plotInd2)
        histOut = plt.hist(orientEntDists, normed=True)
        plt.xlabel("Orientation Entropy")
        plt.ylabel("Probability Density")
        plt.savefig(figureDir + "OrientEnt" +  ".eps")
        #plt.legend()
        plotInd2 += 1

        plt.figure(plotInd2)
        histOut = plt.hist(detectionDists, detectBins, normed=True)
        plt.xlabel("Detection Range (days)")
        plt.ylabel("Probability Density")
        plt.savefig(figureDir + "DetectionRanges" +  ".eps")
        #plt.legend()
        plotInd2 += 1

numConfigGraphs = 10

def computeConfigScalarStats():
    logging.info("Computing configuration model scalar stats")

    graphFileNameBase = resultsDir + "ConfigInfectGraph"
    resultsFileNameBase = resultsDir + "ConfigInfectGraphScalarStats"

    for j in range(numConfigGraphs):
        resultsFileName = resultsFileNameBase + str(j)

        if not os.path.isfile(resultsFileName):
            configGraph = SparseGraph.load(graphFileNameBase + str(j))
            statsArray = graphStats.sequenceScalarStats(configGraph, subgraphIndicesList, slowStats, treeStats=True)
            Util.savePickle(statsArray, resultsFileName, True)
            gc.collect()

    logging.info("All done")

def computeConfigVectorStats():
    #Note: We can make this multithreaded
    logging.info("Computing configuration model vector stats")

    graphFileNameBase = resultsDir + "ConfigInfectGraph"
    resultsFileNameBase = resultsDir + "ConfigInfectGraphVectorStats"

    for j in range(numConfigGraphs):
        resultsFileName = resultsFileNameBase + str(j)

        if not os.path.isfile(resultsFileName):
            configGraph = SparseGraph.load(graphFileNameBase + str(j))
            statsDictList = graphStats.sequenceVectorStats(configGraph, subgraphIndicesList2, eigenStats=False, treeStats=True)
            Util.savePickle(statsDictList, resultsFileName, False)
            gc.collect()

    logging.info("All done")


def plotScalarStats():
    logging.info("Computing scalar stats")
    resultsFileName = resultsDir + "InfectGrowthScalarStats.pkl"


    if saveResults:
        statsArray = graphStats.sequenceScalarStats(sGraph, subgraphIndicesList, treeStats=True)
        Util.savePickle(statsArray, resultsFileName, True)
    else:
        statsArray = Util.loadPickle(resultsFileName)

        global plotInd

        #Output all the results into plots
        #Take the mean of the results over the configuration model graphs
        resultsFileNameBase = resultsDir + "ConfigInfectGraphScalarStats"
        numGraphs = len(subgraphIndicesList)
        configStatsArrays = numpy.zeros((numGraphs, graphStats.getNumStats(), numConfigGraphs))

        for j in range(numConfigGraphs):
            resultsFileName = resultsFileNameBase + str(j)
            configStatsArrays[:, :, j] = Util.loadPickle(resultsFileName)

        configStatsArray = numpy.mean(configStatsArrays, 2)
        configStatsStd = numpy.std(configStatsArrays, 2)

        #Make sure we don't include 0 in the array
        vertexIndex = numpy.argmax(statsArray[:, graphStats.numVerticesIndex] > 0)
        edgeIndex = numpy.argmax(statsArray[:, graphStats.numEdgesIndex] > 0)
        minIndex = numpy.maximum(vertexIndex, edgeIndex)

        def plotRealConfigError(index, styleReal, styleConfig, realLabel, configLabel):
            plt.hold(True)
            plt.plot(absDayList, statsArray[:, index], styleReal, label=realLabel)
            #errors = numpy.c_[configStatsArray[:, index]-configStatsMinArray[:, index] , configStatsMaxArray[:, index]-configStatsArray[:, index]].T
            errors = numpy.c_[configStatsStd[:, index], configStatsStd[:, index]].T
            plt.plot(absDayList, configStatsArray[:, index], styleConfig, label=configLabel)
            plt.errorbar(absDayList, configStatsArray[:, index], errors, linewidth=0, elinewidth=0, label="_nolegend_", ecolor=styleConfig[0])

            xmin, xmax = plt.xlim()
            plt.xlim((0, xmax))
            ymin, ymax = plt.ylim()
            plt.ylim((0, ymax))

        plt.figure(plotInd)
        plt.plot(numpy.log(statsArray[minIndex:, graphStats.numVerticesIndex]), numpy.log(statsArray[minIndex:, graphStats.numEdgesIndex]))
        plt.xlabel("log(|V|)")
        plt.ylabel("log(|E|)")
        plt.savefig(figureDir + "LogVerticesEdgesGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        #plt.plot(absDayList, statsArray[:, graphStats.numTreesIndex], plotStyles3[0], label="Trees Size >= 1")
        #plt.plot(absDayList, statsArray[:, graphStats.numNonSingletonTreesIndex], plotStyles3[1], label="Trees Size >= 2")
        plotRealConfigError(graphStats.numTreesIndex, plotStyles3[0], plotStyles5[0], "Trees size >= 1", "CM trees size >= 1")
        plotRealConfigError(graphStats.numNonSingletonTreesIndex, plotStyles3[0], plotStyles5[0], "Trees size >= 2", "CM trees size >= 2")
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("No. trees")
        plt.legend(loc="upper left")
        plt.savefig(figureDir + "NumTreesGrowth.eps")
        plotInd += 1

        for k in range(len(dayList)):
            day = dayList[k]
            print(str(DateUtils.getDateStrFromDay(day, startYear)) + ": " + str(statsArray[k, graphStats.numTreesIndex]))
            print(str(DateUtils.getDateStrFromDay(day, startYear)) + ": " + str(configStatsArray[k, graphStats.numTreesIndex]))


        #Load stats from a file to get the max tree from its root 
        resultsFilename = resultsDir + "treeSizesDepths.npz"
        file = open(resultsFilename, 'r')
        arrayDict = numpy.load(file)
        statsArray[:, graphStats.maxTreeDepthIndex] = arrayDict["arr_0"]
        statsArray[:, graphStats.maxTreeSizeIndex] = arrayDict["arr_1"]
        statsArray[:, graphStats.secondTreeDepthIndex] = arrayDict["arr_2"]
        statsArray[:, graphStats.secondTreeSizeIndex] = arrayDict["arr_3"]

        plt.figure(plotInd)
        plotRealConfigError(graphStats.maxTreeSizeIndex, plotStyles3[0], plotStyles5[0], "Max tree", "CM max tree")
        plotRealConfigError(graphStats.secondTreeSizeIndex, plotStyles3[1], plotStyles5[1], "2nd tree", "CM 2nd tree")
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Size")
        plt.legend(loc="upper left")
        plt.savefig(figureDir + "MaxTreeGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.maxTreeDepthIndex, plotStyles3[0], plotStyles5[0], "Max tree", "CM max tree")
        plotRealConfigError(graphStats.secondTreeDepthIndex, plotStyles3[1], plotStyles5[1], "2nd tree", "CM 2nd tree")
        #plt.plot(absDayList, statsArray[:, graphStats.maxTreeDepthIndex], plotStyles3[0], absDayList, statsArray[:, graphStats.secondTreeDepthIndex], plotStyles3[1] )
        #plt.plot(absDayList, configStatsArray[:, graphStats.maxTreeDepthIndex], plotStyles4[0], absDayList, configStatsArray[:, graphStats.secondTreeDepthIndex], plotStyles4[1])
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Depth")
        plt.legend(loc="lower right")
        plt.savefig(figureDir + "MaxTreeDepthGrowth.eps")
        
        plotInd += 1


def plotVectorStats():
    #Finally, compute some vector stats at various points in the graph
    logging.info("Computing vector stats")
    global plotInd
    resultsFileName = resultsDir + "InfectGrowthVectorStats.pkl"

    if saveResults:
        statsDictList = graphStats.sequenceVectorStats(sGraph, subgraphIndicesList2, True)
        Util.savePickle(statsDictList, resultsFileName, True)
    else:
        statsDictList = Util.loadPickle(resultsFileName)

        treeSizesDistArray = numpy.zeros((len(dayList2), 3000))
        treeDepthsDistArray = numpy.zeros((len(dayList2), 100))
        numVerticesEdgesArray = numpy.zeros((len(dayList2), 2), numpy.int)
        numVerticesEdgesArray[:, 0] = [len(sgl) for sgl in subgraphIndicesList2]
        numVerticesEdgesArray[:, 1] = [sGraph.subgraph(sgl).getNumEdges() for sgl in subgraphIndicesList2]

        for j in range(len(dayList2)):
            dateStr = (str(DateUtils.getDateStrFromDay(dayList2[j], startYear)))
            logging.info(dateStr)
            statsDict = statsDictList[j]

            degreeDist = statsDict["outDegreeDist"]
            degreeDist = degreeDist/float(numpy.sum(degreeDist))

            maxEigVector = statsDict["maxEigVector"]
            maxEigVector = numpy.flipud(numpy.sort(numpy.abs(maxEigVector)))
            maxEigVector = numpy.log(maxEigVector[maxEigVector>0])

            treeSizesDist = statsDict["treeSizesDist"]
            treeSizesDist = numpy.array(treeSizesDist, numpy.float64)/numpy.sum(treeSizesDist)
            treeSizesDistArray[j, 0:treeSizesDist.shape[0]] = treeSizesDist

            treeDepthsDist = statsDict["treeDepthsDist"]
            #treeDepthsDist = numpy.array(treeDepthsDist, numpy.float64)/numpy.sum(treeDepthsDist)
            treeDepthsDist = numpy.array(treeDepthsDist, numpy.float64)
            treeDepthsDistArray[j, 0:treeDepthsDist.shape[0]] = treeDepthsDist

            plotInd2 = plotInd

            plt.figure(plotInd2)
            plt.plot(numpy.arange(degreeDist.shape[0]), degreeDist, label=dateStr)
            plt.xlabel("Degree")
            plt.ylabel("Probability")
            plt.ylim((0, 0.8))
            plt.legend()
            plt.savefig(figureDir + "DegreeDist" +  ".eps")
            plotInd2 += 1

            plt.figure(plotInd2)
            plt.scatter(numpy.arange(treeSizesDist.shape[0])[treeSizesDist!=0], numpy.log(treeSizesDist[treeSizesDist!=0]), s=30, c=plotStyles2[j][0], label=dateStr)
            plt.xlabel("Size")
            plt.ylabel("log(probability)")
            plt.xlim((0, 125))
            plt.legend()
            plt.savefig(figureDir + "TreeSizeDist" +  ".eps")
            plotInd2 += 1

            plt.figure(plotInd2)
            plt.scatter(numpy.arange(treeDepthsDist.shape[0])[treeDepthsDist!=0], numpy.log(treeDepthsDist[treeDepthsDist!=0]), s=30, c=plotStyles2[j][0], label=dateStr)
            plt.xlabel("Depth")
            plt.ylabel("log(probability)")
            plt.xlim((0, 15))
            plt.legend()
            plt.savefig(figureDir + "TreeDepthDist" +  ".eps")
            plotInd2 += 1

        dateStrList = [DateUtils.getDateStrFromDay(day, startYear) for day in dayList2]
        precision = 4 

        treeSizesDistArray = treeSizesDistArray[:, 0:treeSizesDist.shape[0]]
        nonZeroCols = numpy.sum(treeSizesDistArray, 0)!=0
        print((Latex.array1DToRow(numpy.arange(treeSizesDistArray.shape[1])[nonZeroCols])))
        print((Latex.array2DToRows(treeSizesDistArray[:, nonZeroCols])))

        print("Tree depths")
        treeDepthsDistArray = treeDepthsDistArray[:, 0:treeDepthsDist.shape[0]]
        nonZeroCols = numpy.sum(treeDepthsDistArray, 0)!=0
        print((Latex.array1DToRow(numpy.arange(treeDepthsDistArray.shape[1])[nonZeroCols])))
        print((Latex.array2DToRows(treeDepthsDistArray[:, nonZeroCols])))

        print(numpy.sum(treeDepthsDistArray[:, 0:3], 1))

        print("Edges and verticies")
        print(Latex.listToRow(dateStrList))
        print(Latex.array2DToRows(numVerticesEdgesArray.T, precision))

def plotOtherStats():

    binEdges = numpy.arange(0, 6000, 180)
    binEdges2 = numpy.arange(0, 10000, 365)
    diffDetections = numpy.zeros((len(subgraphIndicesList2), binEdges.shape[0]-1))
    diffDobs = numpy.zeros((len(subgraphIndicesList2), binEdges2.shape[0]-1))

    global plotInd

    for i in range(len(dayList2)):
        dateStr = (str(DateUtils.getDateStrFromDay(dayList2[i], startYear)))
        logging.info(dateStr)
        subgraph = sGraph.subgraph(subgraphIndicesList2[i])
        subVertexArray = subgraph.getVertexList().getVertices()
        edgeIndices = subgraph.getAllEdges()
        
        diffDetections[i, :], binEdges = numpy.histogram(numpy.abs(subVertexArray[edgeIndices[:, 0], detectionIndex] - subVertexArray[edgeIndices[:, 1], detectionIndex]), binEdges)
        diffDetections[i, :] = diffDetections[i, :]/numpy.sum(diffDetections[i, :])

        diffDobs[i, :], binEdges2 = numpy.histogram(numpy.abs(subVertexArray[edgeIndices[:, 0], dobIndex] - subVertexArray[edgeIndices[:, 1], dobIndex]), binEdges2)
        diffDobs[i, :] = diffDobs[i, :]/numpy.sum(diffDobs[i, :])


        plotInd2 = plotInd 

        plt.figure(plotInd2)
        plt.plot((binEdges[1:]+binEdges[0:-1])/2.0, diffDetections[i, :], label=dateStr)
        plt.xlabel("Difference in detection date")
        plt.ylabel("Probability")
        #plt.ylim((0, 0.8))
        plt.legend()
        plt.savefig(figureDir + "DetectionDatesDist" +  ".eps")
        plotInd2 += 1

        plt.figure(plotInd2)
        plt.plot((binEdges2[1:]+binEdges2[0:-1])/2.0, diffDobs[i, :], label=dateStr)
        plt.xlabel("Difference in DoB")
        plt.ylabel("Probability")
        #plt.ylim((0, 0.8))
        plt.legend()
        plt.savefig(figureDir + "BirthDatesDist" +  ".eps")
        plotInd2 += 1
    
def plotEdgeStats():
    logging.info("Computing vertex stats")

    femaleToHeteroMans = numpy.zeros(len(subgraphIndicesList))
    femaleToBis = numpy.zeros(len(subgraphIndicesList))
    heteroManToFemales = numpy.zeros(len(subgraphIndicesList))
    biToFemales = numpy.zeros(len(subgraphIndicesList))
    biToBis = numpy.zeros(len(subgraphIndicesList))
    
    print(len(subgraphIndicesList))
    print(len(dayList))

    for i in range(len(dayList)):
        logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
        subgraphIndices = subgraphIndicesList[i]
        subgraph = sGraph.subgraph(subgraphIndices)
        subVertexArray = subgraph.getVertexList().getVertices()
        edgeIndices = subgraph.getAllEdges()

        femaleToHeteroMans[i] = numpy.sum(numpy.logical_and(numpy.logical_and(subVertexArray[edgeIndices[:, 0], genderIndex]==1, subVertexArray[edgeIndices[:, 1], genderIndex]==0), subVertexArray[edgeIndices[:, 1], orientationIndex]==0))
        femaleToBis[i] = numpy.sum(numpy.logical_and(numpy.logical_and(subVertexArray[edgeIndices[:, 0], genderIndex]==1, subVertexArray[edgeIndices[:, 1], genderIndex]==0), subVertexArray[edgeIndices[:, 1], orientationIndex]==1))
        heteroManToFemales[i] = numpy.sum(numpy.logical_and(numpy.logical_and(subVertexArray[edgeIndices[:, 0], genderIndex]==0, subVertexArray[edgeIndices[:, 0], orientationIndex]==0), subVertexArray[edgeIndices[:, 1], genderIndex]==1))
        biToFemales[i] = numpy.sum(numpy.logical_and(numpy.logical_and(subVertexArray[edgeIndices[:, 0], genderIndex]==0, subVertexArray[edgeIndices[:, 0], orientationIndex]==1), subVertexArray[edgeIndices[:, 1], genderIndex]==1))
        biToBis[i] = numpy.sum(numpy.logical_and(subVertexArray[edgeIndices[:, 0], orientationIndex]==1, subVertexArray[edgeIndices[:, 1], orientationIndex]==1))

    global plotInd

    plt.figure(plotInd)
    plt.plot(absDayList, femaleToHeteroMans, plotStyles2[0], absDayList, femaleToBis, plotStyles2[1], absDayList, heteroManToFemales, plotStyles2[2], absDayList, biToFemales, plotStyles2[3], absDayList, biToBis, plotStyles2[4])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Frequency")
    plt.legend(("Woman to heterosexual man", "Woman to MSM", "Heterosexual man to woman", "MSM to woman", "MSM to MSM"), loc="upper left")
    plt.savefig(figureDir + "EgoAlterGenderOrient.eps")
    plotInd += 1

    for k in range(len(dayList)):
        day = dayList[k]
        print(str(DateUtils.getDateStrFromDay(day, startYear)) + ": " + str(biToFemales[k]))
        print(str(DateUtils.getDateStrFromDay(day, startYear)) + ": " + str(biToBis[k]))

#Some statistics on the largest two trees 
def plotMaxTreesStats():
    biSums1 = []
    heteroSums1 = []
    biSums2 = []
    heteroSums2 = []

    treeDepth1 = [] 
    treeSize1 = []
    treeDepth2 = []
    treeSize2 = [] 

    logging.info("Finding trees")
    trees = sGraph.findTrees()

    maxTree = sGraph.subgraph(trees[0])
    secondTree = sGraph.subgraph(trees[1])

    maxRootIndex = trees[0][numpy.nonzero(sGraph.inDegreeSequence()[trees[0]] == 0)[0]]
    secondRootIndex = trees[1][numpy.nonzero(sGraph.inDegreeSequence()[trees[1]] == 0)[0]]

    for j in range(len(subgraphIndicesList)):
        Util.printIteration(j, 1, len(subgraphIndicesList))
        subgraphIndices = subgraphIndicesList[j]
        subgraphIndices = numpy.array(subgraphIndices)

        currentMaxRootIndex = numpy.nonzero(subgraphIndices == maxRootIndex)[0]
        currentSecondRootIndex = numpy.nonzero(subgraphIndices == secondRootIndex)[0]
        subgraph = sGraph.subgraph(subgraphIndices)

        if currentMaxRootIndex.shape[0] == 1:
            maxTree = subgraph.subgraph(subgraph.depthFirstSearch(currentMaxRootIndex[0]))
        else:
            maxTree = subgraph.subgraph(numpy.array([]))

        if currentSecondRootIndex.shape[0] == 1:
            secondTree = subgraph.subgraph(subgraph.depthFirstSearch(currentSecondRootIndex[0]))
        else:
            secondTree = subgraph.subgraph(numpy.array([]))

        subgraphVertexArray = maxTree.getVertexList().getVertices()
        subgraphVertexArray2 = secondTree.getVertexList().getVertices()
        #Compute proportion of MSM, Male, Female, Hetero
        heteroSums1.append(numpy.sum(subgraphVertexArray[:, orientationIndex]==0))
        biSums1.append(numpy.sum(subgraphVertexArray[:, orientationIndex]==1))

        heteroSums2.append(numpy.sum(subgraphVertexArray2[:, orientationIndex]==0))
        biSums2.append(numpy.sum(subgraphVertexArray2[:, orientationIndex]==1))

        treeDepth1.append(GraphUtils.treeDepth(maxTree))
        treeSize1.append(maxTree.getNumVertices())
        treeDepth2.append(GraphUtils.treeDepth(secondTree))
        treeSize2.append(secondTree.getNumVertices())

    resultsFilename = resultsDir + "treeSizesDepths.npz"
    file = open(resultsFilename, 'w')
    numpy.savez(file, treeDepth1, treeSize1, treeDepth2, treeSize2)

    global plotInd

    plt.figure(plotInd)
    plt.plot(absDayList, heteroSums1, plotStyles3[0], absDayList, biSums1, plotStyles3[1], absDayList, heteroSums2, plotStyles3[2], absDayList, biSums2, plotStyles3[3])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Detections")
    plt.legend(("Max tree heterosexual", "Max tree MSM", "2nd tree heterosexual", "2nd tree MSM"), loc="upper left")
    plt.savefig(figureDir + "MaxTreeOrientGender.eps")
    plotInd += 1

#plotTreeStats()
#plotScalarStats()
plotVectorStats()
#plotOtherStats()
#plotEdgeStats()
#plotMaxTreesStats()
plt.show()

#computeConfigScalarStats()
#computeConfigVectorStats()