import numpy
import logging
import sys 
import matplotlib.pyplot as plt 
from apgl.graph.GraphStatistics import GraphStatistics 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util 
from apgl.predictors.ABCSMC import ABCSMC 
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.sandbox.GraphMatch import GraphMatch
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

plotStyles = ['k-', 'kx-', 'k+-', 'k.-', 'k*-']

outputDir = PathDefaults.getOutputDir() + "viroscopy/"
#resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/"
resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/"
thetaDir = resultsDir + "theta/" 
saveResults = True
graphStats = GraphStatistics()

N = 20 
t = 2

#We plot some stats for the ideal simulated epidemic 
#and those epidemics found using ABC. 

if saveResults:

    thetaArray = ABCSMC.loadThetaArray(N, thetaDir, t)
    logging.debug(thetaArray)
    
    #startDate, endDate, recordStep, printStep, M, targetGraph = HIVModelUtils.toySimulationParams()
    startDate, endDate, recordStep, printStep, M, targetGraph = HIVModelUtils.realSimulationParams()
    
    for i in range(thetaArray.shape[0]): 
        #Compute distances 
        featureInds= numpy.ones(targetGraph.vlist.getNumFeatures(), numpy.bool)
        featureInds[HIVVertices.dobIndex] = False 
        featureInds[HIVVertices.infectionTimeIndex] = False 
        featureInds[HIVVertices.hiddenDegreeIndex] = False 
        featureInds[HIVVertices.stateIndex] = False 
        featureInds = numpy.arange(featureInds.shape[0])[featureInds]        
        
        matcher = GraphMatch("U", 0.5, featureInds=featureInds)
        graphMetrics = HIVGraphMetrics2(targetGraph, 1.0, matcher, float(endDate))        
        
        times, infectedIndices, removedIndices, graph = HIVModelUtils.simulate(thetaArray[i], startDate, endDate, recordStep, printStep, M, graphMetrics)
        stats = HIVModelUtils.generateStatistics(graph, startDate, endDate, recordStep)
    
        print(graphMetrics.dists)
        print(graphMetrics.times)
        break 
    
        resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
        Util.savePickle(stats, resultsFileName)
        

        
        
    #Now save the statistics on the target graph 
    stats = HIVModelUtils.generateStatistics(targetGraph, startDate, endDate, recordStep)
    resultsFileName = outputDir + "IdealStats.pkl"
    Util.savePickle(stats, resultsFileName)
else:
    thetaArray = ABCSMC.loadThetaArray(N, thetaDir, t)
    
    resultsFileName = outputDir + "IdealStats.pkl"
    stats = Util.loadPickle(resultsFileName)  
    times, vertexArray, removedGraphStats = stats 
    
    graphStats = GraphStatistics()
    
    logging.debug(thetaArray.shape)
    #First plot graphs for ideal theta 
    plotInd = 0 
    
    plt.figure(plotInd)
    plt.title("Removed")
    plt.plot(times, vertexArray[:, 0], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Males")
    plt.plot(times, vertexArray[:, 1], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Females")
    plt.plot(times, vertexArray[:, 2], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Hetero")
    plt.plot(times, vertexArray[:, 3], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Bis")
    plt.plot(times, vertexArray[:, 4], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.title("Components")
    plt.plot(times, removedGraphStats[:, graphStats.numComponentsIndex], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of components")
    plotInd += 1

    for i in range(thetaArray.shape[0]): 
        plotInd = 0
        resultsFileName = outputDir + "SimStats" + str(i) + ".pkl"
        stats = Util.loadPickle(resultsFileName)
        
        times, vertexArray, removedGraphStats = stats 

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
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 4], plotStyles[0])
        plotInd += 1
    
        plt.figure(plotInd)
        plt.plot(times, removedGraphStats[:, graphStats.numComponentsIndex], plotStyles[0])
        plotInd += 1

    
    plt.show()
