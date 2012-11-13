
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

plotHIV = True
plotBemol = False
plotCitation = False

BemolSubDir = "Bemol"
HIVSubDir = "HIV"
CitationSubDir = "Citation"

# uncomment data files to read (corresponding curve will be recomputed)
#increasingClustFileName = resultsDir + "IncreasingContrastClustErrors_pmax0.01"


maxPoints = 100             # number of points (dot, square, ...) on curves
startingIteration = 0000    # for HIV, Bemol and Citation data, iteration number of the first data

#==========================================================================
#==========================================================================
#==========================================================================

plotStyles1 = ['k-', 'k--', 'k-.', 'b-', 'b--', 'b-.', 'g-', 'g--', 'g-.', 'r-', 'r--', 'r-.']
plotStyles2 = ['k-', 'k--', 'k-.', 'r-', 'g-', 'b-', 'b--', 'b-.', 'g-', 'g--', 'g-.', 'r-', 'r--', 'r-.']
plotStyles3 = ['b.:', 'bx:', 'b+:', 'bo:', 'b*:', 'bs:']


colorPlotStyles = ['r', 'k', 'g', 'b', 'y', 'm', 'c']
linePlotStyles = ['--', '-', '--', '--']
pointPlotStyles = ['o', 'x', '+', '.']
plotInd = 0

class MyPlot:
    def __init__(self, datasetName, subDirName, k1, k2s, k3s, T=10):
        self.datasetName = datasetName
        self.subDirName = subDirName
        self.measuresList = []
        self.times = []
        self.iterations = []
        self.graphInfosList = []
        
        self.k1 = k1 
        self.k2s = k2s 
        self.k3s = k3s
        self.T = T         
        
        self.methodNames = ["IASC", "Exact", "Ning", "Nystrom"]
        self.labelNames = []
       
    def plotOne(self, data, title, fileNameSuffix, numCol=None, minRow=0, maxRow=None, loc="lower right", xlogscale=False, ylogscale=False, samePlot=False):
        global plotInd
        plt.figure(plotInd)
        for i in range(len(self.labelNames)):
            if len(data[i]) != 0:
                # manage old time reporting
                localNumCol = numCol
                if len(data[i].shape) == 1 or data[i].shape[1] == 1:
                    localNumCol=None
                dataToPrint = data[i][minRow:maxRow,localNumCol]
                #plt.plot(self.iterations[i][minRow:maxRow], dataToPrint, plotStyles1[i])
                #if len(dataToPrint) <= maxPoints:
                #    plt.plot(self.iterations[i][minRow:maxRow], dataToPrint, plotStyles1[i])
                #else:
                #    plt.plot(list(itertools.islice(self.iterations[i][minRow:maxRow],0,None,data[i].size/maxPoints))
                #             , list(itertools.islice(dataToPrint,0,None,data[i].size/maxPoints))
                #             , plotStyles1[i])
                if not samePlot: 
                    plt.plot(self.iterations[i][minRow:maxRow], dataToPrint, plotStyles2[i], label=self.labelNames[i])
                else: 
                    plt.plot(self.iterations[i][minRow:maxRow], dataToPrint, plotStyles1[0], label=self.labelNames[i])
        plt.xlabel("Graph no.")
        plt.ylabel(title)
        if not samePlot: 
            plt.legend(loc=loc, ncol=2)
        if xlogscale:
            plt.xscale('log')
        if ylogscale:
            plt.yscale('log')
        fileName = resultsDir + self.subDirName + "/" + self.datasetName + fileNameSuffix + ".eps"
        logging.debug("Saved " + fileName)
        plt.savefig(fileName)
        plotInd += 1

    def readAll(self):
        for method in self.methodNames:
            
            if method == "Exact": 
                resultsFileName = resultsDir + self.subDirName + "/" + self.datasetName + "ResultsExact_k1=" + str(self.k1) + ".npz"
                self.readFile(resultsFileName) 
                self.labelNames.append("Exact")
            elif method == "IASC": 
                for k2 in self.k2s: 
                    resultsFileName = resultsDir + self.subDirName + "/" + self.datasetName + "ResultsIASC_k1=" + str(self.k1) + "_k2=" + str(k2) + "_T=" + str(self.T) + ".npz"
                    self.readFile(resultsFileName)
                    self.labelNames.append("IASC "+str(k2))
            elif method == "Nystrom": 
                for k3 in self.k3s: 
                    resultsFileName = resultsDir + self.subDirName + "/" + self.datasetName + "ResultsNystrom_k1="+ str(self.k1) + "_k3=" + str(k3) + ".npz"
                    self.readFile(resultsFileName) 
                    self.labelNames.append("Nyst "+str(k3))
            elif method == "Ning": 
                resultsFileName = resultsDir + self.subDirName + "/" + self.datasetName + "ResultsNing_k1=" + str(self.k1) + "_T=" + str(self.T) + ".npz" 
                self.readFile(resultsFileName) 
                self.labelNames.append("Ning")

    def readFile(self, resultsFileName): 
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
            self.times.append(arrayDict["arr_1"].cumsum(0))
            self.graphInfosList.append(arrayDict["arr_2"])
            self.iterations.append(numpy.arange(startingIteration, startingIteration+arrayDict["arr_0"].shape[0]))
            logging.debug("Loaded file " + resultsFileName)

    def plotAll(self):
        self.plotOne(self.measuresList, "Modularity", "Modularities", numCol=0, loc="upper right")
        self.plotOne(self.measuresList, "k-way normalised cut", "KWayNormCut", numCol=1, loc="center right")
#        self.plotOne(self.measuresList, "k-way normalised cut", "KWayNormCut_zoom", numCol=1, maxRow=400, loc="upper right")
        self.plotOne(self.times, "Cumulative computation time (s)", "Time", numCol=0, loc="upper left")
        self.plotOne(self.times, "Cumulative computation time (s)", "Time-log", numCol=0, loc="upper left", xlogscale=False, ylogscale=True)
        self.plotOne(self.graphInfosList, "Nb nodes", "graph_size", numCol=0, loc="lower right", samePlot=True)
        self.plotOne(self.graphInfosList, "Nb connected components", "ConnectedComponents", numCol=1, loc="upper right", samePlot=True)
        
        #print(numpy.c_[numpy.arange(self.graphInfosList[0].shape[0]), self.graphInfosList[0]])

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
    k1 = 25
    k2s = [100, 200, 500]
    k3s = [1000, 1500]    
        
    m = MyPlot("", HIVSubDir, k1, k2s, k3s)
    m.readAll()
    m.plotAll()

if plotBemol:
    k1 = 50
    k2s = [100, 200, 500]
    k3s = [1000, 1500]      
    
    m = MyPlot("", BemolSubDir, k1, k2s, k3s)
    m.readAll()
    m.test()
    m.plotAll()

if plotCitation:
    T = 20
    k1 = 50
    k2s = [50, 100, 200, 500]
    k3s = [500, 1000, 1500]       
    
    m = MyPlot("", CitationSubDir, k1, k2s, k3s, T)
    m.readAll()
    m.plotAll()


#==========================================================================
#==========================================================================
#==========================================================================


#Load IncreasingContrastClustErrors
if 'increasingClustFileName' in locals():
    resIncreasing = {}
    for k2 in [9,18,36,72]:
        file = open(increasingClustFileName + "_nEigen" + str(k2) + ".dat", 'r')
        file.readline()
        resIncreasing[k2] = numpy.loadtxt(file)
    logging.info("Loaded files " + increasingClustFileName)


#==========================================================================
#==========================================================================
#==========================================================================



#plot IncreasingContrast results
numLevel = 3
printedLevel = 2 # in [0, 1, ... , numLevel-1]
startingIteration = 2

if 'increasingClustFileName' in locals():
    iterations = numpy.arange(startingIteration, startingIteration+resIncreasing[9].shape[0])

    fig = plt.figure(plotInd)
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
    
    plt.plot(iterations, resIncreasing[9][:, numLevel*2+printedLevel], plotStyles1[0])
    legend.append("Ning")

    plt.xlim(2, 22)
    plt.xlabel("Graph index")
    plt.ylabel("Rand Index")
    plt.legend(legend)
    plt.savefig(resultsDir + "IncreasingContrastClustErrors_lvl2_paper.eps")


plt.show()

# to run
# python -c "execfile('exp/clusterexp/ProcessClusterResults.py')"
# python3 -c "exec(open('exp/clusterexp/ProcessClusterResults.py').read())"

