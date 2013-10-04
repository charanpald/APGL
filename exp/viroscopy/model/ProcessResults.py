import numpy
import logging
import sys 
import multiprocessing 
import matplotlib.pyplot as plt 
import os
from apgl.graph.GraphStatistics import GraphStatistics 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util 
from apgl.util.Latex import Latex 
from apgl.predictors.ABCSMC import loadThetaArray 
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.sandbox.GraphMatch import GraphMatch
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

processReal = True
saveResults = False 

if processReal: 
    ind = 0 
    resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/theta" + str(ind) + "/"
    outputDir = resultsDir + "stats/"
    startDate, endDates, numRecordSteps, M, targetGraph = HIVModelUtils.realSimulationParams()
    endDate = endDates[ind]
    recordStep = (endDate-startDate)/float(numRecordSteps)
    #endDate += HIVModelUtils.toyTestPeriod
    realTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
else: 
    resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/theta/"
    outputDir = resultsDir + "stats/"
    startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
    endDate += HIVModelUtils.toyTestPeriod
    realTheta, sigmaTheta = HIVModelUtils.toyTheta()

try: 
    os.mkdir(outputDir)
except: 
    pass 

graphStats = GraphStatistics()
N = 20 
t = 0
maxT = 10
minVal = 10 
matchAlpha = 0.2 
breakDist = 1.0

plotStyles = ['k-', 'kx-', 'k+-', 'k.-', 'k*-']

for i in range(maxT): 
    thetaArray, distArray = loadThetaArray(N, resultsDir, i)
    if thetaArray.shape[0] == N: 
        t = i   
    
logging.debug("Using population " + str(t))

#We plot some stats for the ideal simulated epidemic 
#and those epidemics found using ABC. 

def saveStats(args):
    i, theta = args 
    
    featureInds= numpy.ones(targetGraph.vlist.getNumFeatures(), numpy.bool)
    featureInds[HIVVertices.dobIndex] = False 
    featureInds[HIVVertices.infectionTimeIndex] = False 
    featureInds[HIVVertices.hiddenDegreeIndex] = False 
    featureInds[HIVVertices.stateIndex] = False 
    featureInds = numpy.arange(featureInds.shape[0])[featureInds]        
    
    matcher = GraphMatch("PATH", alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
    graphMetrics = HIVGraphMetrics2(targetGraph, breakDist, matcher, float(endDate))        
    times, infectedIndices, removedIndices, graph = HIVModelUtils.simulate(thetaArray[i], startDate, endDate, recordStep, M, graphMetrics)
    times = numpy.arange(startDate, endDate+1, recordStep)
    vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats = HIVModelUtils.generateStatistics(graph, times)
    stats = times, vertexArray, removedGraphStats, graphMetrics.dists, graphMetrics.graphDists, graphMetrics.labelDists
    resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
    Util.savePickle(stats, resultsFileName)

if saveResults:
    thetaArray = loadThetaArray(N, resultsDir, t)[0]
    logging.debug(thetaArray)
    
    paramList = []
    
    for i in range(thetaArray.shape[0]): 
        paramList.append((i, thetaArray[i, :]))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())               
    resultIterator = pool.map(saveStats, paramList)  
    #resultIterator = map(saveStats, paramList)  
    pool.terminate()

    #Now save the statistics on the target graph 
    times = numpy.arange(startDate, endDate+1, recordStep)
    vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats = HIVModelUtils.generateStatistics(targetGraph, times)
    stats = vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats
    resultsFileName = outputDir + "IdealStats.pkl"
    Util.savePickle(stats, resultsFileName)
else:
    realTheta, sigmaTheta = HIVModelUtils.toyTheta()
    thetaArray, distArray = loadThetaArray(N, resultsDir, t)
    print(realTheta)
    print(thetaArray)    
    print(distArray)
    
    meanTable = numpy.c_[realTheta, thetaArray.mean(0)]
    stdTable = numpy.c_[sigmaTheta, thetaArray.std(0)]
    table = Latex.array2DToRows(meanTable, stdTable, precision=4)
    rowNames = ["$\\|\\mathcal{I}_0 \\|$", "$\\alpha$", "$\\gamma$", "$\\beta$", "$\\lambda$",  "$\\sigma$"]
    table = Latex.addRowNames(rowNames, table)
    print(table)

    resultsFileName = outputDir + "IdealStats.pkl"
    stats = Util.loadPickle(resultsFileName)  
    vertexArrayIdeal, infectedIndices, removedIndices, contactGraphStats, removedGraphStats = stats 
    times = numpy.arange(startDate, endDate+1, recordStep)  
    print(times)    
    
    graphStats = GraphStatistics()
    
    #First plot graphs for ideal theta 
    plotInd = 0 
    
    plt.figure(plotInd)
    plt.plot(times, vertexArrayIdeal[:, 0], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Removed")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArrayIdeal[:, 1], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Males")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArrayIdeal[:, 2], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Females")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArrayIdeal[:, 3], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Hetero")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArrayIdeal[:, 4], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Bi")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArrayIdeal[:, 5], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Random detection")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArrayIdeal[:, 6], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Contact Tracing")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, removedGraphStats[:, graphStats.numComponentsIndex], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of components")
    plotInd += 1

    plt.figure(plotInd)
    plt.xlabel("Time (days)")
    plt.ylabel("Distance")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.xlabel("Time (days)")
    plt.ylabel("Graph distance")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.xlabel("Time (days)")
    plt.ylabel("Label distance")
    plotInd += 1
    
    distsArr = []
    detectsArr = []

    for i in range(thetaArray.shape[0]): 
        plotInd = 0
        resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
        stats = Util.loadPickle(resultsFileName)
        
        times, vertexArray, removedGraphStats, dists, graphDists, labelDists = stats 

        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 0], plotStyles[0])
        plotInd += 1 
        detectsArr.append(vertexArray[:, 0])
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 1], plotStyles[0])
        plotInd += 1 

        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 2], plotStyles[0])
        plotInd += 1 
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 3], plotStyles[0])
        plotInd += 1
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 4], plotStyles[0])
        plotInd += 1
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 5], plotStyles[0])
        plotInd += 1
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 6], plotStyles[0])
        plotInd += 1
    
        plt.figure(plotInd)
        plt.plot(times, removedGraphStats[:, graphStats.numComponentsIndex], plotStyles[0])
        plotInd += 1
        

        plt.figure(plotInd)
        distPlotInd = plotInd
        plt.plot(times[1:], dists, plotStyles[0], label="Distance")
        plotInd += 1
        distsArr.append(dists)

        
        plt.figure(plotInd)
        plt.plot(times[1:], graphDists, plotStyles[0])
        plotInd += 1
        
        plt.figure(plotInd)
        plt.plot(times[1:], labelDists, plotStyles[0])
        plotInd += 1

    meanDetects = numpy.array(detectsArr).mean(0)
    stdDetects = numpy.array(detectsArr).std(0)
    plt.figure(plotInd)
    plt.errorbar(times, meanDetects, yerr=stdDetects)  
    plt.plot(times, vertexArrayIdeal[:, 0], "r")
    plotInd += 1
    
    
    meanDists = numpy.array(distsArr).mean(0)
    stdDists = numpy.array(distsArr).std(0)
    plt.figure(plotInd)
    plt.errorbar(times[1:], meanDists, yerr=stdDists) 
    plotInd += 1
    
    plt.show()
