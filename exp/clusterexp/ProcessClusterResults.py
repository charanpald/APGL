
"""
Dump out some graphs 
"""
import os
import sys
import logging
import numpy
import itertools
import matplotlib.pyplot as plt
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)

resultsDir = PathDefaults.getOutputDir() + "cluster/"

plotHIV = False
plotBemol = True
plotCitation = False

BemolSubDir = "cluster_mostrare/Bemol__nbU_1000__nbPurchPerIt_10__startIt_1000__endIt_None/__k1_20__k2_80__k3_80__T_100/"
BemolSubDir = "cluster_mostrare/Bemol__nbU_10000__nbPurchPerIt_10__startIt_30000__endIt_35000/__k1_40__k2_160__k3_160__T_100/"
#BemolSubDir = "Bemol__nbU_1000__nbPurchPerIt_10__endIt_None/__k1_20__k2_80__k3_80__T_100/"
#BemolSubDir = "Bemol100"

# uncomment data files to read (corresponding curve will be recomputed)
#resultsFileName4 = resultsDir + "IncreasingContrastClustErrors_pmax0.001"
#resultsFileName5 = resultsDir + "ThreeClustErrors.dat"

maxPoints = 100             # number of points (dot, square, ...) on curves
startingIteration = 0000    # for HIV, Bemol and Citation data, iteration number of the first data

#==========================================================================
#==========================================================================
#==========================================================================

#methodNames = ["IASC", "Exact", "Modularity", "Ning"]
#labelNames = ["IASC", "Exact", "Modularity", "Ning et al."]
#plotStyles = ['ko--', 'kx-', 'k+--', 'k.--']
#methodNames = ["IASC", "Exact", "Nystrom"]
#labelNames = ["IASC", "Exact", "Nystrom"]
#colorPlotStyles = ['r', 'k', 'b']
#linePlotStyles = ['--', '-', '--']
#pointPlotStyles = ['o', 'x', '.']
methodNames = ["IASC", "Exact", "Ning", "Nystrom"]
labelNames = ["IASC", "Exact", "Ning et al.", "Nystrom"]
colorPlotStyles = ['r', 'k', 'g', 'b']
linePlotStyles = ['--', '-', '--', '--']
pointPlotStyles = ['o', 'x', '+', '.']
plotInd = 0

class MyPlot:
    def __init__(self, datasetName, subDirName):
        self.datasetName = datasetName
        self.subDirName = subDirName
        self.measuresList = []
        self.times = []
        self.iterations = []
        self.graphInfosList = []
       
    def plotOne(self, data, title, fileNameSuffix, numCol=None, minRow=0, maxRow=None, loc="lower right", xlogscale=False, ylogscale=False):
        global plotInd
        plt.figure(plotInd)
        for i in range(len(methodNames)):
            if len(data[i]) != 0:
                # manage old time reporting
                localNumCol = numCol
                if len(data[i].shape) == 1 or data[i].shape[1] == 1:
                    localNumCol=None
                dataToPrint = data[i][minRow:maxRow,localNumCol]
                plt.plot(self.iterations[i][minRow:maxRow], dataToPrint, colorPlotStyles[i] + linePlotStyles[i])
                if len(dataToPrint) <= maxPoints:
                    plt.plot(self.iterations[i][minRow:maxRow], dataToPrint, colorPlotStyles[i] + pointPlotStyles[i])
                else:
                    plt.plot(list(itertools.islice(self.iterations[i][minRow:maxRow],0,None,data[i].size/maxPoints))
                             , list(itertools.islice(dataToPrint,0,None,data[i].size/maxPoints))
                             , colorPlotStyles[i] + pointPlotStyles[i])
                plt.plot(self.iterations[i][minRow], dataToPrint[0], colorPlotStyles[i] + linePlotStyles[i] + pointPlotStyles[i], label=labelNames[i])
        plt.xlabel("Graph no.")
        plt.ylabel(title)
        plt.legend(loc=loc)
        if xlogscale:
            plt.xscale('log')
        if ylogscale:
            plt.yscale('log')
#        fileName = resultsDir + self.subDirName + "/" + self.datasetName + fileNameSuffix + ".eps"
        fileName = resultsDir + self.subDirName + "/" + fileNameSuffix + ".eps"
        print(fileName)
        plt.savefig(fileName)
        plotInd += 1

    def readAll(self):
        for method in methodNames:
#            resultsFileName = resultsDir + self.subDirName + "/" + self.datasetName + "Results" + method +  ".npz"
            resultsFileName = resultsDir + self.subDirName + "/" + "Results" + method +  ".npz"

            try:
                file = open(resultsFileName, 'r')
                arrayDict = numpy.load(file)
            except:
                self.measuresList.append(numpy.array([]))
                self.times.append([])
                self.iterations.append(numpy.array([]))
                self.graphInfosList.append([])
                logging.warning(" file " + resultsFileName + " is empty")
            else:
                self.measuresList.append(arrayDict["arr_0"])
                self.times.append(arrayDict["arr_1"])
                self.graphInfosList.append(arrayDict["arr_2"])
                self.iterations.append(numpy.arange(startingIteration, startingIteration+arrayDict["arr_0"].shape[0]))
                logging.info(" Loaded file " + resultsFileName)

    def plotAll(self):
        self.plotOne(self.measuresList, "Modularity", "Modularities", numCol=0)
        self.plotOne(self.measuresList, "k-way normalised cut", "KWayNormCut", numCol=1)
#        self.plotOne(self.measuresList, "k-way normalised cut", "KWayNormCut_zoom", numCol=1, maxRow=400, loc="upper right")
        self.plotOne(self.times, "Computation time", "Time", numCol=0, loc="upper left")
        self.plotOne(self.times, "Computation time", "Time-log", numCol=0, loc="upper left", xlogscale=False, ylogscale=True)
        self.plotOne(self.graphInfosList, "Nb nodes", "graph_size", numCol=0, loc="lower right")
        self.plotOne(self.graphInfosList, "Nb connected components", "ConnectedComponents", numCol=1, loc="upper right")

    def test(self):
        logging.warning(" test expect IASC being the first method and Exact the second one")
        if len(self.times[0])==0 or len(self.times[1])==0:
            logging.warning(" One of both time data is empty")
            return
        if len(self.times[0].shape) == 1 or self.times[0].shape[1] == 1:
            numColIASC=None
        else:
            numColIASC=0
        if len(self.times[1].shape) == 1 or self.times[1].shape[1] == 1:
            numColExact=None
        else:
            numColExact=0
        for tIASC, tExact, it in zip(self.times[0][:,numColIASC], self.times[1][:,numColExact]
                                     , range(len(self.times[0]))):
            if tIASC > tExact*1.1:
                logging.warning( "IASC took " + str(tIASC/tExact-1)
                                 + " times more computation time than Exact ("
                                 + str(tIASC) + " vs " + str(tExact)
                                 + " at iteration " + str(it) + ")")

if plotHIV:
    m = MyPlot("HIV")
    m.readAll()
    m.plotAll()

if plotBemol:
    m = MyPlot("Bemol", BemolSubDir)
    m.readAll()
    m.test()
    m.plotAll()

if plotCitation:
    m = MyPlot("Citation")
    m.readAll()
    m.plotAll()


#==========================================================================
#==========================================================================
#==========================================================================

#Load static clustering results
if 'resultsFileName2' in locals():
    file = open(resultsFileName2, 'r')
    arrayDict = numpy.load(file)
    logging.info("Loaded file " + resultsFileName2)

    staticModularities = arrayDict["arr_0"]
    staticKwayNormalisedCuts = arrayDict["arr_1"]

#Load IncreasingContrastClustErrors
if 'resultsFileName4' in locals():
    resIncreasing = {}
    for k2 in [9,18,36,72]:
        file = open(resultsFileName4 + "_nEigen" + str(k2) + ".dat", 'r')
        file.readline()
        resIncreasing[k2] = numpy.loadtxt(file)
    logging.info("Loaded files " + resultsFileName4)

#Load 3ClustErrors
if 'resultsFileName5' in locals():
    file = open(resultsFileName5, 'r')
    file.readline()
    res3clust = numpy.loadtxt(file)
    logging.info("Loaded files " + resultsFileName5)

#==========================================================================
#==========================================================================
#==========================================================================

plotStyles1 = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-', 'ks-', 'k-']
plotStyles2 = ['ro--', 'rx--', 'r+--', 'r.--', 'r*--', 'rs--']
plotStyles3 = ['bo:', 'bx:', 'b+:', 'b.:', 'b*:', 'bs:']

#plot IncreasingContrast results
numLevel = 3
printedLevel = 2 # in [0, 1, ... , numLevel-1]
startingIteration = 5

if 'resultsFileName4' in locals():
    iterations = numpy.arange(startingIteration, startingIteration+resIncreasing[9].shape[0])

    plt.figure(plotInd)
    plotInd += 1
    legend = []
    # IASC
    k2s = [9, 18, 72]
    for (k2,i) in zip(k2s, range(len(k2s))):
        plt.plot(iterations, resIncreasing[k2][:, numLevel+printedLevel], plotStyles2[i])
        legend.append("IASC " + str(k2))
    # Nystrom
    k2s = [9, 18, 72]
    for (k2,i) in zip(k2s, range(len(k2s))):
        plt.plot(iterations, resIncreasing[k2][:, numLevel*3+printedLevel], plotStyles3[i])
        legend.append("Nystrom " + str(k2))
    # Exact
    plt.plot(iterations, resIncreasing[9][:, numLevel*0+printedLevel], plotStyles1[6])
    legend.append("Exact")
    plt.ylim(0.15,0.30)
    plt.xlabel("Graph no.")
    plt.ylabel("Rand Index")
    plt.legend(legend)
    plt.savefig(resultsDir + "IncreasingContrastClustErrors_lvl2_paper.eps")

#plot 3clust results
if 'resultsFileName5' in locals():
    numClusters = 3
    startClusterSize = 20
    endClusterSize = 60
    clusterStep = 5

    numVertices = numClusters*numpy.arange(startClusterSize, endClusterSize+1, clusterStep)
    numVertices = numpy.hstack((numVertices,numClusters*numpy.arange(endClusterSize-clusterStep, startClusterSize-1, -clusterStep)))
    ps = numpy.arange(0.05, 0.20, 0.05)
#    self.iterations

    plt.figure(plotInd)
    plotInd += 1
    fig = plt.subplot(111)
    plt.hold(True)
    # for the legend
#    plt.plot(self.iterations, res3clust[:,len(ps)], "k--", self.iterations, res3clust[:, 0], "k-", self.iterations, res3clust[:,2*len(ps)], "k:")
    plt.plot(res3clust[:,len(ps)], "k--", res3clust[:, 0], "k-", res3clust[:,2*len(ps)], "k:")
    for i_p in range(len(ps)):
#        plt.plot(self.iterations, res3clust[:, i_p], plotStyles1[i_p], self.iterations, res3clust[:, len(ps)+i_p], plotStyles2[i_p], self.iterations, res3clust[:, 2*len(ps)+i_p], plotStyles3[i_p])
        plt.plot(res3clust[:, i_p], plotStyles1[i_p], res3clust[:, len(ps)+i_p], plotStyles2[i_p], res3clust[:, 2*len(ps)+i_p], plotStyles3[i_p])
    plt.hold(False)
    plt.xlabel("Number of Vertices")
    from matplotlib.ticker import IndexLocator, FixedFormatter
    tickLocator = IndexLocator(1, 0)
    tickFormatter = FixedFormatter([str(i) for i in numVertices])
    fig.xaxis.set_major_locator(tickLocator)
    fig.xaxis.set_major_formatter(tickFormatter)
#    plt.axis.set_ticklabels(numVertices)
    plt.ylabel("Rand Index")
    plt.legend(("IASC", "Exact", "Ning et al."), loc="upper left")
    plt.savefig(resultsDir + "ThreeClustErrors.eps")

plt.show()

# to run
# python -c "execfile('exp/clusterexp/ProcessClusterResults.py')"
# python3 -c "exec(open('exp/clusterexp/ProcessClusterResults.py').read())"

