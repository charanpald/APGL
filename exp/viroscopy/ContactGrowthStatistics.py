import logging
import sys
import gc 
import numpy
import os.path
import matplotlib.pyplot as plt
from datetime import date
from apgl.util.PathDefaults import PathDefaults
from apgl.util.DateUtils import DateUtils
from apgl.util.Latex import Latex
from apgl.util.Util import Util
from apgl.graph import * 
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.viroscopy.HIVGraphStatistics import HIVGraphStatistics

"""
This script computes some basic statistics on the growing graph. We currently
combine both infection and detection graphs and hence
look at the contact graph. 
"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=150)

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
#sGraph = sGraph.subgraph(range(0, 200))

figureDir = PathDefaults.getOutputDir() + "viroscopy/figures/contact/"
resultsDir = PathDefaults.getOutputDir() + "viroscopy/"

graphStats = GraphStatistics()
statsArray = graphStats.scalarStatistics(sGraph, False)
slowStats = True
saveResults = False

logging.info(sGraph)
logging.info("Number of features: " + str(sGraph.getVertexList().getNumFeatures()))
logging.info("Largest component is " + str(statsArray[graphStats.maxComponentSizeIndex]))
logging.info("Number of components " + str(statsArray[graphStats.numComponentsIndex]))

#sGraph = sGraph.subgraph(components[componentIndex])
vertexArray = sGraph.getVertexList().getVertices()
logging.info("Size of graph we will use: " + str(sGraph.getNumVertices()))

#Some indices
dobIndex = fInds["birthDate"]
detectionIndex = fInds["detectDate"]
deathIndex = fInds["deathDate"]
genderIndex = fInds["gender"]
orientationIndex = fInds["orient"]

ages = vertexArray[:, detectionIndex] - vertexArray[:, dobIndex]
deaths = vertexArray[:, deathIndex] - vertexArray[:, detectionIndex]
detections = vertexArray[:, detectionIndex]

startYear = 1900
daysInYear = 365
daysInMonth = 30
monthStep = 3

#Effective diameter q 
q = 0.9

plotInd = 1
plotStyles = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-']
plotStyles2 = ['k-', 'r-', 'g-', 'b-', 'c-', 'm-']
plotStyleBW = ['k-', 'k--', 'k-.', 'k:']
plotStyles4 = ['r-', 'r--', 'r-.', 'r:']

numConfigGraphs = 10

#Make sure we include all detections
dayList = range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep)
dayList.append(numpy.max(detections))
absDayList = [float(i-numpy.min(detections)) for i in dayList]
subgraphIndicesList = []

for i in dayList:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    subgraphIndicesList.append(subgraphIndices)
    
#Compute the indices list for the vector statistics
dayList2 = [DateUtils.getDayDelta(date(1989, 12, 31), startYear)]
dayList2.append(DateUtils.getDayDelta(date(1993, 12, 31), startYear))
dayList2.append(DateUtils.getDayDelta(date(1997, 12, 31), startYear))
dayList2.append(DateUtils.getDayDelta(date(2001, 12, 31), startYear))
dayList2.append(int(numpy.max(detections)))

subgraphIndicesList2 = []
for i in dayList2:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    subgraphIndicesList2.append(subgraphIndices)

#Locations and labels for years
locs = list(range(0, int(absDayList[-1]), daysInYear*2))
labels = numpy.arange(1986, 2006, 2)

#Some indices
contactIndex = fInds["contactTrace"]
donorIndex = fInds["donor"]
randomTestIndex = fInds["randomTest"]
stdIndex = fInds["STD"]
prisonerIndex = fInds["prisoner"]
doctorIndex = fInds["recommendVisit"]

#The most popular provinces
havanaIndex = fInds["CH"]
villaClaraIndex = fInds["VC"]
pinarIndex = fInds["PR"]
holguinIndex = fInds["HO"]
habanaIndex = fInds["LH"]
sanctiIndex = fInds["SS"]

santiagoIndex = fInds['SC']
camagueyIndex = fInds['CA']

def plotVertexStats():
    #Calculate all vertex statistics
    logging.info("Computing vertex stats")
    
    #Indices
    numContactsIndex = fInds["numContacts"]
    numTestedIndex = fInds["numTested"]
    numPositiveIndex = fInds["numPositive"]

    #Properties of vertex values
    detectionAges = []
    deathAfterInfectAges = []
    deathAges = []
    homoMeans = []

    maleSums = []
    femaleSums = []
    heteroSums = []
    biSums = []

    contactMaleSums = []
    contactFemaleSums = []
    contactHeteroSums = []
    contactBiSums = []

    doctorMaleSums = []
    doctorFemaleSums = []
    doctorHeteroSums = []
    doctorBiSums = []

    contactSums = []
    nonContactSums = []
    donorSums = []
    randomTestSums = []
    stdSums = []
    prisonerSums = []
    recommendSums = []
    #This is: all detections - contact, donor, randomTest, str, recommend
    otherSums = []

    havanaSums = []
    villaClaraSums = []
    pinarSums = []
    holguinSums = []
    habanaSums = []
    sanctiSums = []

    numContactSums = []
    numTestedSums = []
    numPositiveSums = []

    #Total number of sexual contacts 
    numContactMaleSums = []
    numContactFemaleSums = []
    numContactHeteroSums = []
    numContactBiSums = []

    numTestedMaleSums = []
    numTestedFemaleSums = []
    numTestedHeteroSums = []
    numTestedBiSums = []

    numPositiveMaleSums = []
    numPositiveFemaleSums = []
    numPositiveHeteroSums = []
    numPositiveBiSums = []

    propPositiveMaleSums = []
    propPositiveFemaleSums = []
    propPositiveHeteroSums = []
    propPositiveBiSums = []

    numContactVertices = []
    numContactEdges = []
    numInfectEdges = []

    #Mean proportion of degree at end of epidemic 
    meanPropDegree = []
    finalDegreeSequence = numpy.array(sGraph.outDegreeSequence(), numpy.float) 

    degreeOneSums = []
    degreeTwoSums = []
    degreeThreePlusSums = []

    numProvinces = 15
    provinceArray = numpy.zeros((len(subgraphIndicesList), numProvinces))
    m = 0 

    for subgraphIndices in subgraphIndicesList: 
        subgraph = sGraph.subgraph(subgraphIndices)
        infectSubGraph = sGraphInfect.subgraph(subgraphIndices)

        subgraphVertexArray = subgraph.getVertexList().getVertices(range(subgraph.getNumVertices()))

        detectionAges.append(numpy.mean((subgraphVertexArray[:, detectionIndex] - subgraphVertexArray[:, dobIndex]))/daysInYear)
        deathAfterInfectAges.append((numpy.mean(subgraphVertexArray[:, deathIndex] - subgraphVertexArray[:, detectionIndex]))/daysInYear)
        deathAges.append(numpy.mean((subgraphVertexArray[:, deathIndex] - subgraphVertexArray[:, dobIndex]))/daysInYear)
        homoMeans.append(numpy.mean(subgraphVertexArray[:, orientationIndex]))

        nonContactSums.append(subgraphVertexArray.shape[0] - numpy.sum(subgraphVertexArray[:, contactIndex]))
        contactSums.append(numpy.sum(subgraphVertexArray[:, contactIndex]))
        donorSums.append(numpy.sum(subgraphVertexArray[:, donorIndex]))
        randomTestSums.append(numpy.sum(subgraphVertexArray[:, randomTestIndex]))
        stdSums.append(numpy.sum(subgraphVertexArray[:, stdIndex]))
        prisonerSums.append(numpy.sum(subgraphVertexArray[:, prisonerIndex]))
        recommendSums.append(numpy.sum(subgraphVertexArray[:, doctorIndex]))
        otherSums.append(subgraphVertexArray.shape[0] - numpy.sum(subgraphVertexArray[:, [contactIndex, donorIndex, randomTestIndex, stdIndex, doctorIndex]]))

        heteroSums.append(numpy.sum(subgraphVertexArray[:, orientationIndex]==0))
        biSums.append(numpy.sum(subgraphVertexArray[:, orientationIndex]==1))

        femaleSums.append(numpy.sum(subgraphVertexArray[:, genderIndex]==1))
        maleSums.append(numpy.sum(subgraphVertexArray[:, genderIndex]==0))

        contactHeteroSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, orientationIndex]==0, subgraphVertexArray[:, contactIndex])))
        contactBiSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, orientationIndex]==1, subgraphVertexArray[:, contactIndex])))
        contactFemaleSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, genderIndex]==1, subgraphVertexArray[:, contactIndex])))
        contactMaleSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, genderIndex]==0, subgraphVertexArray[:, contactIndex])))

        doctorHeteroSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, orientationIndex]==0, subgraphVertexArray[:, doctorIndex])))
        doctorBiSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, orientationIndex]==1, subgraphVertexArray[:, doctorIndex])))
        doctorFemaleSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, genderIndex]==1, subgraphVertexArray[:, doctorIndex])))
        doctorMaleSums.append(numpy.sum(numpy.logical_and(subgraphVertexArray[:, genderIndex]==0, subgraphVertexArray[:, doctorIndex])))

        havanaSums.append(numpy.sum(subgraphVertexArray[:, havanaIndex]==1))
        villaClaraSums.append(numpy.sum(subgraphVertexArray[:, villaClaraIndex]==1))
        pinarSums.append(numpy.sum(subgraphVertexArray[:, pinarIndex]==1))
        holguinSums.append(numpy.sum(subgraphVertexArray[:, holguinIndex]==1))
        habanaSums.append(numpy.sum(subgraphVertexArray[:, habanaIndex]==1))
        sanctiSums.append(numpy.sum(subgraphVertexArray[:, sanctiIndex]==1))

        numContactSums.append(numpy.mean(subgraphVertexArray[:, numContactsIndex]))
        numTestedSums.append(numpy.mean(subgraphVertexArray[:, numTestedIndex]))
        numPositiveSums.append(numpy.mean(subgraphVertexArray[:, numPositiveIndex]))

        numContactMaleSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, genderIndex]==0, numContactsIndex]))
        numContactFemaleSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, genderIndex]==1, numContactsIndex]))
        numContactHeteroSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, orientationIndex]==0, numContactsIndex]))
        numContactBiSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, orientationIndex]==1, numContactsIndex]))

        numTestedMaleSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, genderIndex]==0, numTestedIndex]))
        numTestedFemaleSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, genderIndex]==1, numTestedIndex]))
        numTestedHeteroSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, orientationIndex]==0, numTestedIndex]))
        numTestedBiSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, orientationIndex]==1, numTestedIndex]))

        numPositiveMaleSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, genderIndex]==0, numPositiveIndex]))
        numPositiveFemaleSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, genderIndex]==1, numPositiveIndex]))
        numPositiveHeteroSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, orientationIndex]==0, numPositiveIndex]))
        numPositiveBiSums.append(numpy.mean(subgraphVertexArray[subgraphVertexArray[:, orientationIndex]==1, numPositiveIndex]))

        propPositiveMaleSums.append(numPositiveMaleSums[m]/float(numTestedMaleSums[m]))
        propPositiveFemaleSums.append(numPositiveFemaleSums[m]/float(numTestedFemaleSums[m]))
        propPositiveHeteroSums.append(numPositiveHeteroSums[m]/float(numTestedHeteroSums[m]))
        propPositiveBiSums.append(numPositiveBiSums[m]/float(numTestedMaleSums[m]))

        numContactVertices.append(subgraph.getNumVertices())
        numContactEdges.append(subgraph.getNumEdges())
        numInfectEdges.append(infectSubGraph.getNumEdges())

        nonZeroInds = finalDegreeSequence[subgraphIndices]!=0
        propDegrees = numpy.mean(subgraph.outDegreeSequence()[nonZeroInds]/finalDegreeSequence[subgraphIndices][nonZeroInds])
        meanPropDegree.append(numpy.mean(propDegrees)) 

        degreeOneSums.append(numpy.sum(subgraph.outDegreeSequence()==1))
        degreeTwoSums.append(numpy.sum(subgraph.outDegreeSequence()==2))
        degreeThreePlusSums.append(numpy.sum(subgraph.outDegreeSequence()>=3))

        provinceArray[m, :] = numpy.sum(subgraphVertexArray[:, fInds["CA"]:fInds['VC']+1], 0)
        m += 1 

    #Save some of the results for the ABC work
    numStats = 2 
    vertexStatsArray = numpy.zeros((len(subgraphIndicesList), numStats))
    vertexStatsArray[:, 0] = numpy.array(biSums)
    vertexStatsArray[:, 1] = numpy.array(heteroSums)

    resultsFileName = resultsDir + "ContactGrowthVertexStats.pkl"
    Util.savePickle(vertexStatsArray, resultsFileName)

    global plotInd 

    plt.figure(plotInd)
    plt.plot(absDayList, detectionAges)
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Detection Age (years)")
    plt.savefig(figureDir + "DetectionMeansGrowth.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, heteroSums, 'k-', absDayList, biSums, 'k--', absDayList, femaleSums, 'k-.', absDayList, maleSums, 'k:')
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Detections")
    plt.legend(("Heterosexual", "MSM", "Female", "Male"), loc="upper left")
    plt.savefig(figureDir + "OrientationGenderGrowth.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, contactHeteroSums, 'k-', absDayList, contactBiSums, 'k--', absDayList, contactFemaleSums, 'k-.', absDayList, contactMaleSums, 'k:')
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Contact tracing detections")
    plt.legend(("Heterosexual", "MSM", "Female", "Male"), loc="upper left")
    plt.savefig(figureDir + "OrientationGenderContact.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, doctorHeteroSums, 'k-', absDayList, doctorBiSums, 'k--', absDayList, doctorFemaleSums, 'k-.', absDayList, doctorMaleSums, 'k:')
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Doctor recommendation detections")
    plt.legend(("Heterosexual", "MSM", "Female", "Male"), loc="upper left")
    plt.savefig(figureDir + "OrientationGenderDoctor.eps")
    plotInd += 1



    #Plot all the provinces 
    plt.figure(plotInd)
    plt.hold(True)
    for k in range(provinceArray.shape[1]):
        plt.plot(absDayList, provinceArray[:, k], label=str(k))
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Detections")
    plt.legend(loc="upper left")
    plotInd += 1 

    #Plot of detection types
    plt.figure(plotInd)
    plt.plot(absDayList, contactSums, plotStyles2[0], absDayList, donorSums, plotStyles2[1], absDayList, randomTestSums, plotStyles2[2], absDayList, stdSums, plotStyles2[3], absDayList, otherSums, plotStyles2[4], absDayList, recommendSums, plotStyles2[5])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Detections")
    plt.legend(("Contact tracing", "Blood donation", "Random test", "STD", "Other test", "Doctor recommendation"), loc="upper left")
    plt.savefig(figureDir + "DetectionGrowth.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, numContactSums, plotStyleBW[0], absDayList, numTestedSums, plotStyleBW[1], absDayList, numPositiveSums, plotStyleBW[2])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Contacts")
    plt.legend(("No. contacts", "No. tested", "No. positive"), loc="center left")
    plt.savefig(figureDir + "ContactsGrowth.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, numContactHeteroSums, plotStyleBW[0], absDayList, numContactBiSums, plotStyleBW[1], absDayList, numContactFemaleSums, plotStyleBW[2], absDayList, numContactMaleSums, plotStyleBW[3])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Total contacts")
    plt.legend(("Heterosexual", "MSM", "Female", "Male"), loc="upper right")
    plt.savefig(figureDir + "ContactsGrowthOrientGen.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, numTestedHeteroSums, plotStyleBW[0], absDayList, numTestedBiSums, plotStyleBW[1], absDayList, numTestedFemaleSums, plotStyleBW[2], absDayList, numTestedMaleSums, plotStyleBW[3])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Tested contacts")
    plt.legend(("Heterosexual", "MSM", "Female", "Male"), loc="upper right")
    plt.savefig(figureDir + "TestedGrowthOrientGen.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, numPositiveHeteroSums, plotStyleBW[0], absDayList, numPositiveBiSums, plotStyleBW[1], absDayList, numPositiveFemaleSums, plotStyleBW[2], absDayList, numPositiveMaleSums, plotStyleBW[3])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Positive contacts")
    plt.legend(("Heterosexual", "MSM", "Female", "Male"), loc="upper right")
    plt.savefig(figureDir + "PositiveGrowthOrientGen.eps")
    plotInd += 1

    #Proportion positive versus tested
    plt.figure(plotInd)
    plt.plot(absDayList, propPositiveHeteroSums, plotStyleBW[0], absDayList, propPositiveBiSums, plotStyleBW[1], absDayList, propPositiveFemaleSums, plotStyleBW[2], absDayList, propPositiveMaleSums, plotStyleBW[3])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Proportion positive contacts")
    plt.legend(("Heterosexual", "MSM", "Female", "Male"), loc="upper right")
    plt.savefig(figureDir + "PercentPositiveGrowthOrientGen.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.hold(True)
    plt.plot(absDayList, havanaSums, plotStyles2[0])
    plt.plot(absDayList, villaClaraSums, plotStyles2[1])
    plt.plot(absDayList, pinarSums, plotStyles2[2])
    plt.plot(absDayList, holguinSums, plotStyles2[3])
    plt.plot(absDayList, habanaSums, plotStyles2[4])
    plt.plot(absDayList, sanctiSums, plotStyles2[5])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Detections")
    plt.legend(("Havana City", "Villa Clara", "Pinar del Rio", "Holguin", "La Habana", "Sancti Spiritus"), loc="upper left")
    plt.savefig(figureDir + "ProvinceGrowth.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, numContactVertices, plotStyleBW[0], absDayList, numContactEdges, plotStyleBW[1], absDayList, numInfectEdges, plotStyleBW[2])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Vertices/edges")
    plt.legend(("Contact vertices", "Contact edges", "Infect edges"), loc="upper left")
    plt.savefig(figureDir + "VerticesEdges.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, meanPropDegree, plotStyleBW[0])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Proportion of final degree")
    plt.savefig(figureDir + "MeanPropDegree.eps")
    plotInd += 1

    plt.figure(plotInd)
    plt.plot(absDayList, degreeOneSums, plotStyleBW[0], absDayList, degreeTwoSums, plotStyleBW[1], absDayList, degreeThreePlusSums, plotStyleBW[2])
    plt.xticks(locs, labels)
    plt.xlabel("Year")
    plt.ylabel("Detections")
    plt.legend(("Degree = 1", "Degree = 2", "Degree >= 3"), loc="upper left")
    plotInd += 1

    #Print a table of interesting stats
    results = numpy.array([havanaSums])
    results = numpy.r_[results, numpy.array([villaClaraSums])]
    results = numpy.r_[results, numpy.array([pinarSums])]
    results = numpy.r_[results, numpy.array([holguinSums])]
    results = numpy.r_[results, numpy.array([habanaSums])]
    results = numpy.r_[results, numpy.array([sanctiSums])]

    print(Latex.listToRow(["Havana City", "Villa Clara", "Pinar del Rio", "Holguin", "La Habana", "Sancti Spiritus"]))
    print("\\hline")
    for i in range(0, len(dayList), 4):
        day = dayList[i]
        print(str(DateUtils.getDateStrFromDay(day, startYear)) + " & " + Latex.array1DToRow(results[:, i].T) + "\\\\")

    results = numpy.array([heteroSums])
    results = numpy.r_[results, numpy.array([biSums])]
    results = numpy.r_[results, numpy.array([femaleSums])]
    results = numpy.r_[results, numpy.array([maleSums])]

    print("\n\n")
    print(Latex.listToRow(["Heterosexual", "MSM", "Female", "Male"]))
    print("\\hline")
    for i in range(0, len(dayList), 4):
        day = dayList[i]
        print(str(DateUtils.getDateStrFromDay(day, startYear)) + " & " + Latex.array1DToRow(results[:, i].T) + "\\\\")


def computeConfigScalarStats():
    logging.info("Computing configuration model scalar stats")

    graphFileNameBase = resultsDir + "ConfigGraph"
    resultsFileNameBase = resultsDir + "ConfigGraphScalarStats"
    #graphStats.useFloydWarshall = True

    for j in range(numConfigGraphs):
        resultsFileName = resultsFileNameBase + str(j)

        if not os.path.isfile(resultsFileName):
            configGraph = SparseGraph.load(graphFileNameBase + str(j))
            statsArray = graphStats.sequenceScalarStats(configGraph, subgraphIndicesList, slowStats)
            Util.savePickle(statsArray, resultsFileName, True)
            gc.collect()

    logging.info("All done")

def computeConfigVectorStats():
    #Note: We can make this multithreaded 
    logging.info("Computing configuration model vector stats")

    graphFileNameBase = resultsDir + "ConfigGraph"
    resultsFileNameBase = resultsDir + "ConfigGraphVectorStats"

    for j in range(numConfigGraphs):
        resultsFileName = resultsFileNameBase + str(j)

        if not os.path.isfile(resultsFileName):
            configGraph = SparseGraph.load(graphFileNameBase + str(j))
            statsDictList = graphStats.sequenceVectorStats(configGraph, subgraphIndicesList2, eigenStats=False)
            Util.savePickle(statsDictList, resultsFileName, False)
            gc.collect()

    logging.info("All done")

def plotScalarStats():
    logging.info("Computing scalar stats")

    resultsFileName = resultsDir + "ContactGrowthScalarStats.pkl"

    if saveResults:
        statsArray = graphStats.sequenceScalarStats(sGraph, subgraphIndicesList, slowStats)
        Util.savePickle(statsArray, resultsFileName, True)

        #Now compute statistics on the configuration graphs 
    else:
        statsArray = Util.loadPickle(resultsFileName)

        #Take the mean of the results over the configuration model graphs
        resultsFileNameBase = resultsDir + "ConfigGraphScalarStats"
        numGraphs = len(subgraphIndicesList)
        #configStatsArrays = numpy.zeros((numGraphs, graphStats.getNumStats(), numConfigGraphs))
        configStatsArrays = numpy.zeros((numGraphs, graphStats.getNumStats()-2, numConfigGraphs))

        for j in range(numConfigGraphs):
            resultsFileName = resultsFileNameBase + str(j)
            configStatsArrays[:, :, j] = Util.loadPickle(resultsFileName)

        configStatsArray = numpy.mean(configStatsArrays, 2)
        configStatsStd =  numpy.std(configStatsArrays, 2)
        global plotInd

        def plotRealConfigError(index, styleReal, styleConfig, realLabel, configLabel):
            plt.hold(True)
            plt.plot(absDayList, statsArray[:, index], styleReal, label=realLabel)
            #errors = numpy.c_[configStatsArray[:, index]-configStatsMinArray[:, index] , configStatsMaxArray[:, index]-configStatsArray[:, index]].T
            errors = numpy.c_[configStatsStd[:, index], configStatsStd[:, index]].T
            plt.plot(absDayList, configStatsArray[:, index], styleConfig, label=configLabel)
            plt.errorbar(absDayList, configStatsArray[:, index], errors, linewidth=0, elinewidth=1, label="_nolegend_", ecolor="red")

            xmin, xmax = plt.xlim()
            plt.xlim((0, xmax))
            ymin, ymax = plt.ylim()
            plt.ylim((0, ymax))


        #Output all the results into plots
        plt.figure(plotInd)
        plt.hold(True)
        plotRealConfigError(graphStats.maxComponentSizeIndex, plotStyleBW[0], plotStyles4[0], "Max comp. vertices", "CM max comp. vertices")
        plotRealConfigError(graphStats.maxComponentEdgesIndex, plotStyleBW[1], plotStyles4[1], "Max comp. edges", "CM max comp. edges")
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("No. vertices/edges")
        plt.legend(loc="upper left")
        plt.savefig(figureDir + "MaxComponentSizeGrowth.eps")
        plotInd += 1

        for k in range(len(dayList)):
            day = dayList[k]
            print(str(DateUtils.getDateStrFromDay(day, startYear)) + ": " + str(statsArray[k, graphStats.maxComponentEdgesIndex]))
            #print(str(DateUtils.getDateStrFromDay(day, startYear)) + ": " + str(configStatsArray[k, graphStats.numComponentsIndex]))

        plt.figure(plotInd)
        plotRealConfigError(graphStats.numComponentsIndex, plotStyleBW[0], plotStyles4[0], "Size >= 1", "CM size >= 1")
        plotRealConfigError(graphStats.numNonSingletonComponentsIndex, plotStyleBW[1], plotStyles4[1], "Size >= 2", "CM size >= 2")
        plotRealConfigError(graphStats.numTriOrMoreComponentsIndex, plotStyleBW[2], plotStyles4[2], "Size >= 3", "CM size >= 3")

        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("No. components")
        plt.legend(loc="upper left")
        plt.savefig(figureDir + "NumComponentsGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.meanComponentSizeIndex, plotStyleBW[0], plotStyles4[0], "Real graph", "CM")
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Mean component size")
        plt.legend(loc="lower right")
        plt.savefig(figureDir + "MeanComponentSizeGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.diameterIndex, plotStyleBW[0], plotStyles4[0], "Real graph", "CM")
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Max component diameter")
        plt.legend(loc="lower right")
        plt.savefig(figureDir + "MaxComponentDiameterGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.effectiveDiameterIndex, plotStyleBW[0], plotStyles4[0], "Real graph", "CM")
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Effective diameter")
        plt.legend(loc="lower right")
        plt.savefig(figureDir + "MaxComponentEffDiameterGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.meanDegreeIndex, plotStyleBW[0], plotStyles4[0], "All vertices", "CM all vertices")
        plotRealConfigError(graphStats.maxCompMeanDegreeIndex, plotStyleBW[1], plotStyles4[1], "Max component", "CM max component")
        #plt.plot(absDayList, statsArray[:, graphStats.meanDegreeIndex], plotStyleBW[0], absDayList, statsArray[:, graphStats.maxCompMeanDegreeIndex], plotStyleBW[1], absDayList, configStatsArray[:, graphStats.meanDegreeIndex], plotStyles4[0], absDayList, configStatsArray[:, graphStats.maxCompMeanDegreeIndex], plotStyles4[1])
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Mean degree")
        plt.legend(loc="lower right")
        plt.savefig(figureDir + "MeanDegrees.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.densityIndex, plotStyleBW[0], plotStyles4[0], "Real Graph", "Config Model")
        #plt.plot(absDayList, statsArray[:, graphStats.densityIndex], plotStyleBW[0], absDayList, configStatsArray[:, graphStats.densityIndex], plotStyles4[0])
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(figureDir + "DensityGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plt.plot(absDayList, statsArray[:, graphStats.powerLawIndex], plotStyleBW[0])
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Alpha")
        plt.savefig(figureDir + "PowerLawGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.geodesicDistanceIndex, plotStyleBW[0], plotStyles4[0], "Real Graph", "Config Model")
        #plt.plot(absDayList, statsArray[:, graphStats.geodesicDistanceIndex], plotStyleBW[0], absDayList, configStatsArray[:, graphStats.geodesicDistanceIndex], plotStyles4[0])
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Geodesic distance")
        plt.legend(loc="lower right")
        plt.savefig(figureDir + "GeodesicGrowth.eps")
        plotInd += 1

        plt.figure(plotInd)
        plotRealConfigError(graphStats.harmonicGeoDistanceIndex, plotStyleBW[0], plotStyles4[0], "Real Graph", "Config Model")
        #plt.plot(absDayList, statsArray[:, graphStats.harmonicGeoDistanceIndex], plotStyleBW[0], absDayList, configStatsArray[:, graphStats.harmonicGeoDistanceIndex], plotStyles4[0])
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Mean harmonic geodesic distance")
        plt.legend(loc="upper right")
        plt.savefig(figureDir + "HarmonicGeodesicGrowth.eps")
        plotInd += 1

        #print(statsArray[:, graphStats.harmonicGeoDistanceIndex])

        plt.figure(plotInd)
        plotRealConfigError(graphStats.geodesicDistMaxCompIndex, plotStyleBW[0], plotStyles4[0], "Real graph", "Config model")
        #plt.plot(absDayList, statsArray[:, graphStats.geodesicDistMaxCompIndex], plotStyleBW[0], absDayList, configStatsArray[:, graphStats.geodesicDistMaxCompIndex], plotStyles4[0])
        plt.xticks(locs, labels)
        plt.xlabel("Year")
        plt.ylabel("Max component mean geodesic distance")
        plt.legend(loc="lower right")
        plt.savefig(figureDir + "MaxCompGeodesicGrowth.eps")
        plotInd += 1

        #Find the number of edges in the infection graph
        resultsFileName = resultsDir + "InfectGrowthScalarStats.pkl"
        infectStatsArray = Util.loadPickle(resultsFileName)

        #Make sure we don't include 0 in the array
        vertexIndex = numpy.argmax(statsArray[:, graphStats.numVerticesIndex] > 0)
        edgeIndex = numpy.argmax(infectStatsArray[:, graphStats.numEdgesIndex] > 0)
        minIndex = numpy.maximum(vertexIndex, edgeIndex)

        plt.figure(plotInd)
        plt.plot(numpy.log(statsArray[minIndex:, graphStats.numVerticesIndex]), numpy.log(statsArray[minIndex:, graphStats.numEdgesIndex]), plotStyleBW[0])
        plt.plot(numpy.log(infectStatsArray[minIndex:, graphStats.numVerticesIndex]), numpy.log(infectStatsArray[minIndex:, graphStats.numEdgesIndex]), plotStyleBW[1])
        plt.plot(numpy.log(statsArray[minIndex:, graphStats.maxComponentSizeIndex]), numpy.log(statsArray[minIndex:, graphStats.maxComponentEdgesIndex]), plotStyleBW[2])
        plt.xlabel("log(|V|)")
        plt.ylabel("log(|E|)/log(|D|)")
        plt.legend(("Contact graph", "Infection graph", "Max component"), loc="upper left")
        plt.savefig(figureDir + "LogVerticesEdgesGrowth.eps")
        plotInd += 1

    results = statsArray[:, graphStats.effectiveDiameterIndex] 
    results = numpy.c_[results, configStatsArray[:, graphStats.effectiveDiameterIndex]]
    results = numpy.c_[results, statsArray[:, graphStats.geodesicDistMaxCompIndex]]
    results = numpy.c_[results, configStatsArray[:, graphStats.geodesicDistMaxCompIndex]]
    configStatsArray

    print("\n\n")
    print(Latex.listToRow(["Diameter", "CM Diameter", "Mean Geodesic", "CM Mean Geodesic"]))
    print("\\hline")
    for i in range(0, len(dayList), 4):
        day = dayList[i]
        print(str(DateUtils.getDateStrFromDay(day, startYear)) + " & " + Latex.array1DToRow(results[i, :]) + "\\\\")



def plotVectorStats():
    #Finally, compute some vector stats at various points in the graph
    logging.info("Computing vector stats")
    global plotInd
    resultsFileName = resultsDir + "ContactGrowthVectorStats.pkl"

    if saveResults:
        statsDictList = graphStats.sequenceVectorStats(sGraph, subgraphIndicesList2)
        Util.savePickle(statsDictList, resultsFileName, False)
    else:
        statsDictList = Util.loadPickle(resultsFileName)

        #Load up configuration model results
        configStatsDictList = []
        resultsFileNameBase = resultsDir + "ConfigGraphVectorStats"

        for j in range(numConfigGraphs):
            resultsFileName = resultsFileNameBase + str(j)
            configStatsDictList.append(Util.loadPickle(resultsFileName))

        #Now need to take mean of 1st element of list
        meanConfigStatsDictList = configStatsDictList[0]
        for i in range(len(configStatsDictList[0])):
            for k in range(1, numConfigGraphs):
                for key in configStatsDictList[k][i].keys():
                    if configStatsDictList[k][i][key].shape[0] > meanConfigStatsDictList[i][key].shape[0]:
                        meanConfigStatsDictList[i][key] = numpy.r_[meanConfigStatsDictList[i][key], numpy.zeros(configStatsDictList[k][i][key].shape[0] - meanConfigStatsDictList[i][key].shape[0])]
                    elif configStatsDictList[k][i][key].shape[0] < meanConfigStatsDictList[i][key].shape[0]:
                        configStatsDictList[k][i][key] = numpy.r_[configStatsDictList[k][i][key], numpy.zeros(meanConfigStatsDictList[i][key].shape[0] - configStatsDictList[k][i][key].shape[0])]

                    meanConfigStatsDictList[i][key] += configStatsDictList[k][i][key]

            for key in configStatsDictList[0][i].keys():
                meanConfigStatsDictList[i][key] = meanConfigStatsDictList[i][key]/numConfigGraphs


        triangleDistArray = numpy.zeros((len(dayList2), 100))
        configTriangleDistArray = numpy.zeros((len(dayList2), 100))
        hopPlotArray = numpy.zeros((len(dayList2), 27))
        configHopPlotArray = numpy.zeros((len(dayList2), 30))
        componentsDistArray = numpy.zeros((len(dayList2), 3000))
        configComponentsDistArray = numpy.zeros((len(dayList2), 3000))
        numVerticesEdgesArray = numpy.zeros((len(dayList2), 2), numpy.int)
        numVerticesEdgesArray[:, 0] = [len(sgl) for sgl in subgraphIndicesList2]
        numVerticesEdgesArray[:, 1] = [sGraph.subgraph(sgl).getNumEdges() for sgl in subgraphIndicesList2]

        binWidths = numpy.arange(0, 0.50, 0.05)
        eigVectorDists = numpy.zeros((len(dayList2), binWidths.shape[0]-1), numpy.int)

        femaleSums = numpy.zeros(len(dayList2))
        maleSums = numpy.zeros(len(dayList2))
        heteroSums = numpy.zeros(len(dayList2))
        biSums = numpy.zeros(len(dayList2))

        contactSums = numpy.zeros(len(dayList2))
        nonContactSums = numpy.zeros(len(dayList2))
        donorSums = numpy.zeros(len(dayList2))
        randomTestSums = numpy.zeros(len(dayList2))
        stdSums = numpy.zeros(len(dayList2))
        prisonerSums = numpy.zeros(len(dayList2))
        recommendSums = numpy.zeros(len(dayList2))
        
        meanAges = numpy.zeros(len(dayList2))
        degrees = numpy.zeros((len(dayList2), 20))

        provinces = numpy.zeros((len(dayList2), 15))

        havanaSums = numpy.zeros(len(dayList2))
        villaClaraSums = numpy.zeros(len(dayList2))
        pinarSums = numpy.zeros(len(dayList2))
        holguinSums = numpy.zeros(len(dayList2))
        habanaSums = numpy.zeros(len(dayList2))
        sanctiSums = numpy.zeros(len(dayList2))

        meanDegrees = numpy.zeros(len(dayList2))
        stdDegrees = numpy.zeros(len(dayList2))

        #Note that death has a lot of missing values
        for j in range(len(dayList2)):
            dateStr = (str(DateUtils.getDateStrFromDay(dayList2[j], startYear)))
            logging.info(dateStr)
            statsDict = statsDictList[j]
            configStatsDict = meanConfigStatsDictList[j]

            degreeDist = statsDict["outDegreeDist"]
            degreeDist = degreeDist/float(numpy.sum(degreeDist))
            #Note that degree distribution for configuration graph will be identical 

            eigenDist = statsDict["eigenDist"]
            eigenDist = numpy.log(eigenDist[eigenDist>=10**-1])
            #configEigenDist = configStatsDict["eigenDist"]
            #configEigenDist = numpy.log(configEigenDist[configEigenDist>=10**-1])

            hopCount = statsDict["hopCount"]
            hopCount = numpy.log10(hopCount)
            hopPlotArray[j, 0:hopCount.shape[0]] = hopCount
            configHopCount = configStatsDict["hopCount"]
            configHopCount = numpy.log10(configHopCount)
            #configHopPlotArray[j, 0:configHopCount.shape[0]] = configHopCount

            triangleDist = statsDict["triangleDist"]
            #triangleDist = numpy.array(triangleDist, numpy.float64)/numpy.sum(triangleDist)
            triangleDist = numpy.array(triangleDist, numpy.float64)
            triangleDistArray[j, 0:triangleDist.shape[0]] = triangleDist
            configTriangleDist = configStatsDict["triangleDist"]
            configTriangleDist = numpy.array(configTriangleDist, numpy.float64)/numpy.sum(configTriangleDist)
            configTriangleDistArray[j, 0:configTriangleDist.shape[0]] = configTriangleDist

            maxEigVector = statsDict["maxEigVector"]
            eigenvectorInds = numpy.flipud(numpy.argsort(numpy.abs(maxEigVector)))
            top10eigenvectorInds = eigenvectorInds[0:numpy.round(eigenvectorInds.shape[0]/10.0)]
            maxEigVector = numpy.abs(maxEigVector[eigenvectorInds])
            #print(maxEigVector)
            eigVectorDists[j, :] = numpy.histogram(maxEigVector, binWidths)[0]

            componentsDist = statsDict["componentsDist"]
            componentsDist = numpy.array(componentsDist, numpy.float64)/numpy.sum(componentsDist)
            componentsDistArray[j, 0:componentsDist.shape[0]] = componentsDist
            configComponentsDist = configStatsDict["componentsDist"]
            configComponentsDist = numpy.array(configComponentsDist, numpy.float64)/numpy.sum(configComponentsDist)
            configComponentsDistArray[j, 0:configComponentsDist.shape[0]] = configComponentsDist

            plotInd2 = plotInd

            plt.figure(plotInd2)
            plt.plot(numpy.arange(degreeDist.shape[0]), degreeDist, plotStyles2[j], label=dateStr)
            plt.xlabel("Degree")
            plt.ylabel("Probability")
            plt.ylim((0, 0.5))
            plt.savefig(figureDir + "DegreeDist" +  ".eps")
            plt.legend()
            plotInd2 += 1

            """
            plt.figure(plotInd2)
            plt.plot(numpy.arange(eigenDist.shape[0]), eigenDist, label=dateStr)
            plt.xlabel("Eigenvalue rank")
            plt.ylabel("log(Eigenvalue)")
            plt.savefig(figureDir + "EigenDist" +  ".eps")
            plt.legend()
            plotInd2 += 1
            """

            #How does kleinberg do the hop plots 
            plt.figure(plotInd2)
            plt.plot(numpy.arange(hopCount.shape[0]), hopCount, plotStyles[j], label=dateStr)
            plt.xlabel("k")
            plt.ylabel("log10(pairs)")
            plt.ylim( (2.5, 7) )
            plt.legend(loc="lower right")
            plt.savefig(figureDir + "HopCount" + ".eps")
            plotInd2 += 1
            
            plt.figure(plotInd2)
            plt.plot(numpy.arange(maxEigVector.shape[0]), maxEigVector, plotStyles2[j], label=dateStr)
            plt.xlabel("Rank")
            plt.ylabel("log(eigenvector coefficient)")
            plt.savefig(figureDir + "MaxEigVector" +  ".eps")
            plt.legend()
            plotInd2 += 1

            #Compute some information the 10% most central vertices
            
            subgraphIndices = numpy.nonzero(detections <= dayList2[j])[0]
            subgraph = sGraph.subgraph(subgraphIndices)
            subgraphVertexArray = subgraph.getVertexList().getVertices()

            femaleSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, genderIndex]==1)
            maleSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, genderIndex]==0)
            heteroSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, orientationIndex]==0)
            biSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, orientationIndex]==1)

            contactSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, contactIndex])
            donorSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, donorIndex])
            randomTestSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, randomTestIndex])
            stdSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, stdIndex])
            prisonerSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, prisonerIndex])
            recommendSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, doctorIndex])

            meanAges[j] = numpy.mean(subgraphVertexArray[top10eigenvectorInds, detectionIndex] - subgraphVertexArray[top10eigenvectorInds, dobIndex])/daysInYear

            havanaSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, havanaIndex])
            villaClaraSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, villaClaraIndex])
            pinarSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, pinarIndex])
            holguinSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, holguinIndex])
            habanaSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, habanaIndex])
            sanctiSums[j] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, sanctiIndex])

            provinces[j, :] = numpy.sum(subgraphVertexArray[top10eigenvectorInds, 22:37], 0)

            ddist = numpy.bincount(subgraph.outDegreeSequence()[top10eigenvectorInds])
            degrees[j, 0:ddist.shape[0]] = numpy.array(ddist, numpy.float)/numpy.sum(ddist)

            meanDegrees[j] = numpy.mean(subgraph.outDegreeSequence()[top10eigenvectorInds])
            stdDegrees[j] = numpy.std(subgraph.outDegreeSequence()[top10eigenvectorInds])


            plt.figure(plotInd2)
            plt.plot(numpy.arange(degrees[j, :].shape[0]), degrees[j, :], plotStyles2[j], label=dateStr)
            plt.xlabel("Degree")
            plt.ylabel("Probability")
            #plt.ylim((0, 0.5))
            plt.savefig(figureDir + "DegreeDistCentral" +  ".eps")
            plt.legend()
            plotInd2 += 1

        precision = 4
        dateStrList = [DateUtils.getDateStrFromDay(day, startYear) for day in dayList2]

        print("Hop counts")
        print(Latex.listToRow(dateStrList))
        print(Latex.array2DToRows(hopPlotArray.T))

        print("\nHop counts for configuration graphs")
        print(Latex.listToRow(dateStrList))
        print(Latex.array2DToRows(configHopPlotArray.T))

        print("\n\nEdges and vertices")
        print((Latex.listToRow(dateStrList)))
        print((Latex.array2DToRows(numVerticesEdgesArray.T, precision)))

        print("\n\nEigenvector distribution")
        print((Latex.array1DToRow(binWidths[1:]) + "\\\\"))
        print((Latex.array2DToRows(eigVectorDists)))

        print("\n\nDistribution of component sizes")
        componentsDistArray = componentsDistArray[:, 0:componentsDist.shape[0]]
        nonZeroCols = numpy.sum(componentsDistArray, 0)!=0
        componentsDistArray = numpy.r_[numpy.array([numpy.arange(componentsDistArray.shape[1])[nonZeroCols]]), componentsDistArray[:, nonZeroCols]]
        print((Latex.listToRow(dateStrList)))
        print((Latex.array2DToRows(componentsDistArray.T, precision)))

        print("\n\nDistribution of component sizes in configuration graphs")
        configComponentsDistArray = configComponentsDistArray[:, 0:configComponentsDist.shape[0]]
        nonZeroCols = numpy.sum(configComponentsDistArray, 0)!=0
        configComponentsDistArray = numpy.r_[numpy.array([numpy.arange(configComponentsDistArray.shape[1])[nonZeroCols]]), configComponentsDistArray[:, nonZeroCols]]
        print((Latex.listToRow(dateStrList)))
        print((Latex.array2DToRows(configComponentsDistArray.T, precision)))

        print("\n\nDistribution of triangle participations")
        triangleDistArray = triangleDistArray[:, 0:triangleDist.shape[0]]
        nonZeroCols = numpy.sum(triangleDistArray, 0)!=0
        triangleDistArray = numpy.r_[numpy.array([numpy.arange(triangleDistArray.shape[1])[nonZeroCols]])/2, triangleDistArray[:, nonZeroCols]]
        print((Latex.listToRow(dateStrList)))
        print((Latex.array2DToRows(triangleDistArray.T, precision)))

        configTriangleDistArray = configTriangleDistArray[:, 0:configTriangleDist.shape[0]]
        nonZeroCols = numpy.sum(configTriangleDistArray, 0)!=0
        configTriangleDistArray = numpy.r_[numpy.array([numpy.arange(configTriangleDistArray.shape[1])[nonZeroCols]])/2, configTriangleDistArray[:, nonZeroCols]]
        configTriangleDistArray = numpy.c_[configTriangleDistArray, numpy.zeros((configTriangleDistArray.shape[0], triangleDistArray.shape[1]-configTriangleDistArray.shape[1]))]

        print("\n\nDistribution of central vertices")
        print((Latex.listToRow(dateStrList)))
        subgraphSizes = numpy.array(maleSums + femaleSums, numpy.float)
        print("Female & " + Latex.array1DToRow(femaleSums*100/subgraphSizes, 1) + "\\\\")
        print("Male & " + Latex.array1DToRow(maleSums*100/subgraphSizes, 1) + "\\\\")
        print("\hline")
        print("Heterosexual & " + Latex.array1DToRow(heteroSums*100/subgraphSizes, 1) + "\\\\")
        print("Bisexual & " + Latex.array1DToRow(biSums*100/subgraphSizes, 1) + "\\\\")
        print("\hline")
        print("Contact traced & " + Latex.array1DToRow(contactSums*100/subgraphSizes, 1) + "\\\\")
        print("Blood donor & " + Latex.array1DToRow(donorSums*100/subgraphSizes, 1) + "\\\\")
        print("RandomTest & " + Latex.array1DToRow(randomTestSums*100/subgraphSizes, 1) + "\\\\")
        print("STD & " + Latex.array1DToRow(stdSums*100/subgraphSizes, 1) + "\\\\")
        print("Prisoner & " + Latex.array1DToRow(prisonerSums*100/subgraphSizes, 1) + "\\\\")
        print("Doctor recommendation & " + Latex.array1DToRow(recommendSums*100/subgraphSizes, 1) + "\\\\")
        print("\hline")
        print("Mean ages (years) & " + Latex.array1DToRow(meanAges, 2) + "\\\\")
        print("\hline")
        print("Holguin & " + Latex.array1DToRow(holguinSums*100/subgraphSizes, 1) + "\\\\")
        print("La Habana & " + Latex.array1DToRow(habanaSums*100/subgraphSizes, 1) + "\\\\")
        print("Havana City & " + Latex.array1DToRow(havanaSums*100/subgraphSizes, 1) + "\\\\")
        print("Pinar del Rio & " + Latex.array1DToRow(pinarSums*100/subgraphSizes, 1) + "\\\\")
        print("Sancti Spiritus & " + Latex.array1DToRow(sanctiSums*100/subgraphSizes, 1) + "\\\\")
        print("Villa Clara & " + Latex.array1DToRow(villaClaraSums*100/subgraphSizes, 1) + "\\\\")
        print("\hline")
        print("Mean degrees & " + Latex.array1DToRow(meanDegrees, 2) + "\\\\")
        print("Std degrees & " + Latex.array1DToRow(stdDegrees, 2) + "\\\\")
        
        print("\n\nProvinces")
        print(Latex.array2DToRows(provinces))

        print("\n\nDegree distribution")
        print(Latex.array2DToRows(degrees))



def plotOtherStats():
    #Let's look at geodesic distances in subgraphs and communities
    logging.info("Computing other stats")

    resultsFileName = resultsDir + "ContactGrowthOtherStats.pkl"
    hivGraphStats = HIVGraphStatistics(fInds)

    if saveResults:
        statsArray = hivGraphStats.sequenceScalarStats(sGraph, subgraphIndicesList)
        #statsArray["dayList"] = absDayList
        Util.savePickle(statsArray, resultsFileName, True)
    else:
        statsArray = Util.loadPickle(resultsFileName)
        #Just load the harmonic geodesic distances of the full graph 
        resultsFileName = resultsDir + "ContactGrowthScalarStats.pkl"
        statsArray2 = Util.loadPickle(resultsFileName)

        global plotInd

        msmGeodesic = statsArray[:, hivGraphStats.msmGeodesicIndex]
        msmGeodesic[msmGeodesic < 0] = 0
        msmGeodesic[msmGeodesic == float('inf')] = 0

        #Output all the results into plots
        plt.figure(plotInd)
        plt.plot(absDayList, msmGeodesic, 'k-', absDayList, statsArray[:, hivGraphStats.mostConnectedGeodesicIndex], 'k--')
        plt.xticks(locs, labels)
        #plt.ylim([0, 0.1])
        plt.xlabel("Year")
        plt.ylabel("Mean harmonic geodesic distance")
        plt.legend(("MSM individuals", "Top 10% degree"), loc="upper right")
        plt.savefig(figureDir + "MSM10Geodesic" + ".eps")
        plotInd += 1


        plt.figure(plotInd)
        plt.plot(absDayList, statsArray2[:, graphStats.harmonicGeoDistanceIndex], 'k-', absDayList, statsArray[:, hivGraphStats.menSubgraphGeodesicIndex], 'k--')
        plt.xticks(locs, labels)
        plt.ylim([0, 200.0])
        plt.xlabel("Year")
        plt.ylabel("Mean harmonic geodesic distance")
        plt.legend(("All individuals", "Men subgraph"), loc="upper right")
        plt.savefig(figureDir + "MenSubgraphGeodesic" + ".eps")
        plotInd += 1


#plotVertexStats()
plotScalarStats()
#plotVectorStats()
#plotOtherStats()
plt.show()

#computeConfigScalarStats()
#computeConfigVectorStats()

"""
Probability of adding node based on degree - try to find how we can generate data.
Mean Time between first and last infection for each person
"""