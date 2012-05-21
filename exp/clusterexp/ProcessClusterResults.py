
"""
Dump out some graphs 
"""
import sys
import logging
import numpy
import matplotlib.pyplot as plt
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)

resultsDir = PathDefaults.getOutputDir() + "cluster/"

# uncomment data files to read (corresponding curve will be recomputed)
plotHIV = False
plotBemol = True
plotCitation = False

#resultsFileName4 = resultsDir + "IncreasingContrastClustErrors_pmax0.01"
#resultsFileName5 = resultsDir + "ThreeClustErrors.dat"

#==========================================================================
#==========================================================================
#==========================================================================

#methodNames = ["IASC", "Exact", "Modularity", "Ning"]
#labelNames = ["IASC", "Exact", "Modularity", "Ning et al."]
#plotStyles = ['ko--', 'kx-', 'k+--', 'k.--']
methodNames = ["IASC", "Exact", "Nystrom"]
labelNames = ["IASC", "Exact", "Nystrom"]
plotStyles = ['ko--', 'kx-', 'k.--']
#methodNames = ["IASC", "Exact", "Ning", "Nystrom"]
#labelNames = ["IASC", "Exact", "Ning et al.", "Nystrom"]
#plotStyles = ['ko--', 'kx-', 'k+--', 'k.--']
plotInd = 0

class MyPlot:
    def __init__(self, datasetName):
        self.datasetName = datasetName
        self.measuresList = []
        self.times = []
        self.iterations = []
        self.graphInfosList = []
       
    def plotOne(self, data, title, fileNameSuffix, numCol=None, maxRow=None, loc="lower right"):
        global plotInd
        plt.figure(plotInd)
        for i in range(len(methodNames)):
            if not self.measuresList[i].size == 0:
                plt.plot(self.iterations[i][:maxRow], data[i][:maxRow,numCol], plotStyles[i], label=labelNames[i])
        plt.xlabel("Graph no.")
        plt.ylabel(title)
        plt.legend(loc=loc)
        print(resultsDir + self.datasetName + fileNameSuffix + ".eps")
        plt.savefig(resultsDir + self.datasetName + fileNameSuffix + ".eps")
        plotInd += 1

    def readAll(self):
        for method in methodNames:
            resultsFileName = resultsDir + self.datasetName + "Results" + method +  ".npz"

            try:
                file = open(resultsFileName, 'r')
                arrayDict = numpy.load(file)
                self.measuresList.append(arrayDict["arr_0"])
                self.times.append(arrayDict["arr_1"])
                self.graphInfosList.append(arrayDict["arr_2"])
                self.iterations.append(numpy.arange(arrayDict["arr_0"].shape[0]))
                logging.info(" Loaded file " + resultsFileName)
            except:
                self.measuresList.append(numpy.array([]))
                self.times.append([])
                self.iterations.append(numpy.array([]))
                self.graphInfosList.append([])
                logging.warning(" file " + resultsFileName + " is empty")

    def plotAll(self):
        self.plotOne(self.measuresList, "Modularity", "Modularities", numCol=0)
        self.plotOne(self.measuresList, "k-way normalised cut", "KWayNormCut", numCol=1)
        self.plotOne(self.measuresList, "k-way normalised cut", "KWayNormCut_zoom", numCol=1, maxRow=400, loc="upper right")
        self.plotOne(self.times, "Computation time", "Time", loc="upper left")
        self.plotOne(self.graphInfosList, "Nb nodes", "graph_size", numCol=0, loc="lower right")
        self.plotOne(self.graphInfosList, "Nb connected components", "ConnectedComponents", numCol=1, loc="upper right")


if plotHIV:
    m = MyPlot("HIV")
    m.readAll()
    m.plotAll()

if plotBemol:
    m = MyPlot("Bemol")
    m.readAll()
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
	startingIteration = 3
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

plotStyles1 = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-', 'ks-']
plotStyles2 = ['ko--', 'kx--', 'k+--', 'k.--', 'k*--', 'ks--']
plotStyles3 = ['ko:', 'kx:', 'k+:', 'k.:', 'k*:', 'ks:']

#plot IncreasingContrast results
if 'resultsFileName4' in locals():
	self.iterations = numpy.arange(startingIteration, startingIteration+resIncreasing[9].shape[0])

	plt.figure(4)
	plt.plot(self.iterations, resIncreasing[9][:, 5], plotStyles2[2], self.iterations, resIncreasing[18][:, 5], plotStyles2[3], self.iterations, resIncreasing[36][:, 5], plotStyles2[4], self.iterations, resIncreasing[72][:, 5], plotStyles2[5], self.iterations, resIncreasing[9][:, 2], plotStyles1[0], self.iterations, resIncreasing[9][:, 8], plotStyles3[1])
	plt.xlabel("Graph no.")
	plt.ylabel("Rand Index")
	plt.legend(("IASC 9", "IASC 18", "IASC 36", "IASC 72", "Exact", "Ning et al."), loc="upper right")
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
#	self.iterations

	plt.figure(5)
	fig = plt.subplot(111)
	plt.hold(True)
	# for the legend
#	plt.plot(self.iterations, res3clust[:,len(ps)], "k--", self.iterations, res3clust[:, 0], "k-", self.iterations, res3clust[:,2*len(ps)], "k:")
	plt.plot(res3clust[:,len(ps)], "k--", res3clust[:, 0], "k-", res3clust[:,2*len(ps)], "k:")
	for i_p in range(len(ps)):
#		plt.plot(self.iterations, res3clust[:, i_p], plotStyles1[i_p], self.iterations, res3clust[:, len(ps)+i_p], plotStyles2[i_p], self.iterations, res3clust[:, 2*len(ps)+i_p], plotStyles3[i_p])
		plt.plot(res3clust[:, i_p], plotStyles1[i_p], res3clust[:, len(ps)+i_p], plotStyles2[i_p], res3clust[:, 2*len(ps)+i_p], plotStyles3[i_p])
	plt.hold(False)
	plt.xlabel("Number of Vertices")
	from matplotlib.ticker import IndexLocator, FixedFormatter
	tickLocator = IndexLocator(1, 0)
	tickFormatter = FixedFormatter([str(i) for i in numVertices])
	fig.xaxis.set_major_locator(tickLocator)
	fig.xaxis.set_major_formatter(tickFormatter)
#	plt.axis.set_ticklabels(numVertices)
	plt.ylabel("Rand Index")
	plt.legend(("IASC", "Exact", "Ning et al."), loc="upper left")
	plt.savefig(resultsDir + "ThreeClustErrors.eps")

plt.show()

# to run
# python -c "execfile('exp/clusterexp/ProcessClusterResults.py')"
# python3 -c "exec(open('exp/clusterexp/ProcessClusterResults.py').read())"

