
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
plotHIV = True
plotBemol = True
plotCitation = False

#resultsFileName4 = resultsDir + "IncreasingContrastClustErrors_pmax0.01"
#resultsFileName5 = resultsDir + "ThreeClustErrors.dat"

plotStyles = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-', 'ks-']
plotStyles2 = ['ko--', 'kx--', 'k+--', 'k.--', 'k*--', 'ks--']
plotStyles3 = ['ko:', 'kx:', 'k+:', 'k.:', 'k*:', 'ks:']

#==========================================================================
#==========================================================================
#==========================================================================

#methodNames = ["IASC", "Exact", "Modularity", "Ning"]
#labelNames = ["IASC", "Exact", "Modularity", "Ning et al."]
methodNames = ["IASC", "Exact", "Ning", "Nystrom"]
labelNames = ["IASC", "Exact", "Ning et al.", "Nystrom"]
plotInd = 0 

def plotDataset(dataset):
    global plotInd
    measuresList = []
    iterations = [];
    
    for method in methodNames:
        resultsFileName = resultsDir + dataset + "Results" + method +  ".npz"

        try:
            file = open(resultsFileName, 'r')
            arrayDict = numpy.load(file)
            measuresList.append(arrayDict["arr_0"])
            iterations.append(numpy.arange(arrayDict["arr_0"].shape[0]))
            logging.info(" Loaded file " + resultsFileName)
        except:
            measuresList.append(numpy.array([]))
            iterations.append(numpy.array([]))
            logging.warning(" file " + resultsFileName + " is empty")

    plt.figure(plotInd)
    for i in range(len(methodNames)):
        if not measuresList[i].size == 0:
            plt.plot(iterations[i], measuresList[i][:, 0], plotStyles2[i], label=labelNames[i])
    plt.xlabel("Graph no.")
    plt.ylabel("Modularity")
    plt.legend(loc="lower right")
    plt.savefig(resultsDir + dataset + "Modularities.eps")
    plotInd += 1

    plt.figure(plotInd)
    for i in range(len(methodNames)):
        if not measuresList[i].size == 0:
            plt.plot(iterations[i], measuresList[i][:, 1], plotStyles2[i], label=labelNames[i])
    plt.xlabel("Graph no.")
    plt.ylabel("k-way normalised cut")
    plt.legend(loc="lower right")
    plt.savefig(resultsDir + dataset + "KWayNormCut.eps")
    plotInd += 1

if plotHIV:
    plotDataset("HIV")

if plotBemol:
    plotDataset("Bemol")

if plotCitation:
    plotDataset("Citation")

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

#plot IncreasingContrast results
if 'resultsFileName4' in locals():
	iterations = numpy.arange(startingIteration, startingIteration+resIncreasing[9].shape[0])

	plt.figure(4)
	plt.plot(iterations, resIncreasing[9][:, 5], plotStyles2[2], iterations, resIncreasing[18][:, 5], plotStyles2[3], iterations, resIncreasing[36][:, 5], plotStyles2[4], iterations, resIncreasing[72][:, 5], plotStyles2[5], iterations, resIncreasing[9][:, 2], plotStyles[0], iterations, resIncreasing[9][:, 8], plotStyles3[1])
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
#	iterations

	plt.figure(5)
	fig = plt.subplot(111)
	plt.hold(True)
	# for the legend
#	plt.plot(iterations, res3clust[:,len(ps)], "k--", iterations, res3clust[:, 0], "k-", iterations, res3clust[:,2*len(ps)], "k:")
	plt.plot(res3clust[:,len(ps)], "k--", res3clust[:, 0], "k-", res3clust[:,2*len(ps)], "k:")
	for i_p in range(len(ps)):
#		plt.plot(iterations, res3clust[:, i_p], plotStyles[i_p], iterations, res3clust[:, len(ps)+i_p], plotStyles2[i_p], iterations, res3clust[:, 2*len(ps)+i_p], plotStyles3[i_p])
		plt.plot(res3clust[:, i_p], plotStyles[i_p], res3clust[:, len(ps)+i_p], plotStyles2[i_p], res3clust[:, 2*len(ps)+i_p], plotStyles3[i_p])
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
# python -c "execfile('apgl/clusterexp/ProcessClusterResults.py')"

