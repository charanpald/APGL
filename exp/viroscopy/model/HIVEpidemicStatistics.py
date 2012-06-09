import numpy
import logging
import sys 
import matplotlib.pyplot as plt 
from apgl.graph.GraphStatistics import GraphStatistics 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util 
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

plotStyles = ['k-', 'kx-', 'k+-', 'k.-', 'k*-']

outputDir = PathDefaults.getOutputDir() + "viroscopy/"
resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
figureDir = PathDefaults.getOutputDir() + "viroscopy/figures/toyExample/"
saveResults = False
graphStats = GraphStatistics()

#We plot some stats for the ideal simulated epidemic 
#and those epidemics found using ABC. 

if saveResults:
    thetaFileName =  resultsDir + "ThetaDistSimulated.pkl"
    thetaArray = Util.loadPickle(thetaFileName)
    thetaArray = numpy.array(thetaArray)
    
    logging.debug(thetaArray.shape)
    
    for i in range(thetaArray.shape[0]): 
        stats = HIVModelUtils.generateStatistics(thetaArray[i])
    
        resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
        Util.savePickle(stats, resultsFileName)
        
    #Now save the true theta 
    theta = HIVModelUtils.defaultTheta()
    stats = HIVModelUtils.generateStatistics(theta)
    resultsFileName = outputDir + "SimStatsIdeal.pkl"
    Util.savePickle(stats, resultsFileName)
else:
    thetaFileName =  resultsDir + "ThetaDistSimulated.pkl"
    thetaArray = Util.loadPickle(thetaFileName)
    thetaArray = numpy.array(thetaArray)
    
    theta = HIVModelUtils.defaultTheta()
    stats = HIVModelUtils.generateStatistics(theta)
    resultsFileName = outputDir + "SimStatsIdeal.pkl"
    stats = Util.loadPickle(resultsFileName)  
    times, vertexArray, infectedGraphStats, removedGraphStats = stats 
    
    logging.debug(thetaArray.shape)
    #First plot graphs for ideal theta 
    plotInd = 0 
    
    plt.figure(plotInd)
    plt.title("Infected")
    plt.plot(times, vertexArray[:, 0], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Removed")
    plt.plot(times, vertexArray[:, 1], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Males")
    plt.plot(times, vertexArray[:, 2], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Females")
    plt.plot(times, vertexArray[:, 3], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    vertexArray[:, 1] += numpy.array(vertexArray[:, 1]==0, numpy.float)
    coverage = vertexArray[:, 1]/numpy.array(vertexArray[:, 1]+vertexArray[:, 0], numpy.float)
    plt.figure(plotInd)
    plt.plot(times, coverage, 'r')
    plt.xlabel("Time (days)")
    plt.ylabel("Coverage")
    plotInd += 1
    
    for i in range(thetaArray.shape[0]): 
        plotInd = 0
        resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
        stats = Util.loadPickle(resultsFileName)
        
        times, vertexArray, infectedGraphStats, removedGraphStats = stats 

        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 0], plotStyles[0])
        plotInd += 1 
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 1], plotStyles[0])
        plotInd += 1 

        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 2], plotStyles[0])
        plotInd += 1 
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 3], plotStyles[0])
        plotInd += 1
        
        vertexArray[:, 1] += numpy.array(vertexArray[:, 1]==0, numpy.float)
        coverage = vertexArray[:, 1]/numpy.array(vertexArray[:, 1]+vertexArray[:, 0], numpy.float)
        plt.figure(plotInd)
        plt.plot(times, coverage, plotStyles[0])

    plt.show()
