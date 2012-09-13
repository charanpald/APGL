import numpy
import logging
import sys 
import multiprocessing 
import matplotlib.pyplot as plt 
from apgl.graph.GraphStatistics import GraphStatistics 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util 
from apgl.util.Latex import Latex 
from apgl.predictors.ABCSMC import ABCSMC, loadThetaArray 
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.sandbox.GraphMatch import GraphMatch
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)

plotStyles = ['k-', 'kx-', 'k+-', 'k.-', 'k*-']

#resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/theta/"
#startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
#endDate += HIVModelUtils.toyTestPeriod

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/theta0/"
startDate, endDates, numRecordSteps, M, targetGraph = HIVModelUtils.realSimulationParams()
endDate = endDates[0]
endDate += HIVModelUtils.realTestPeriod
recordStep = (endDate-startDate)/float(numRecordSteps)

saveResults = False 
graphStats = GraphStatistics()

N = 6 
t = 10

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
    
    matcher = GraphMatch("PATH", alpha=0.5, featureInds=featureInds, useWeightM=False)
    graphMetrics = HIVGraphMetrics2(targetGraph, 1.0, matcher, float(endDate))        
    
    times, infectedIndices, removedIndices, graph = HIVModelUtils.simulate(thetaArray[i], startDate, endDate, recordStep, M, graphMetrics)
    times, vertexArray, removedGraphStats = HIVModelUtils.generateStatistics(graph, startDate, endDate, recordStep)

    stats = times, vertexArray, removedGraphStats, graphMetrics.dists
    
    resultsFileName = resultsDir + "SimStats" + str(i) + ".pkl"
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
    stats = HIVModelUtils.generateStatistics(targetGraph, startDate, endDate, recordStep)
    resultsFileName = resultsDir + "IdealStats.pkl"
    Util.savePickle(stats, resultsFileName)
else:
    #for i in range(t): 
    #    thetaArray = loadThetaArray(N, resultsDir, i)[0]

    realTheta, sigmaTheta = HIVModelUtils.toyTheta()
    thetaArray = loadThetaArray(N, resultsDir, t)[0]
    print(realTheta)
    print(thetaArray)    
    
    
    meanTable = numpy.c_[realTheta, thetaArray.mean(0)]
    stdTable = numpy.c_[sigmaTheta, thetaArray.std(0)]
    table = Latex.array2DToRows(meanTable, stdTable, precision=4)
    rowNames = ["$\\|\\mathcal{I}_0 \\|$", "$\\rho_B$", "$\\alpha$", "$C$", "$\\gamma$", "$\\beta$", "$\\kappa_{max}$", "$\\lambda_H$", "$\\lambda_B$", "$\\sigma_{WM}$",  "$\\sigma_{MW}$","$\\sigma_{MB}$"]
    table = Latex.addRowNames(rowNames, table)
    print(table)

    
    resultsFileName = resultsDir + "IdealStats.pkl"
    stats = Util.loadPickle(resultsFileName)  
    times, vertexArray, removedGraphStats = stats 
    
    times2 = numpy.arange(startDate, endDate+1, recordStep)  
    times2 = times2[1:]
    
    graphStats = GraphStatistics()
    
    #First plot graphs for ideal theta 
    plotInd = 0 
    
    plt.figure(plotInd)
    plt.plot(times, vertexArray[:, 0], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Removed")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArray[:, 1], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Males")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArray[:, 2], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Females")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArray[:, 3], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Hetero")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArray[:, 4], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Bi")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArray[:, 5], "r")
    plt.xlabel("Time (days)")
    plt.ylabel("Random detection")
    plotInd += 1
    
    plt.figure(plotInd)
    plt.plot(times, vertexArray[:, 6], "r")
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
    plt.ylabel("Graph distance")
    plotInd += 1
    
    meanDists = []

    for i in range(thetaArray.shape[0]): 
        plotInd = 0
        resultsFileName = resultsDir + "SimStats" + str(i) + ".pkl"
        stats = Util.loadPickle(resultsFileName)
        
        times, vertexArray, removedGraphStats, dists = stats 

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
        plt.plot(times, vertexArray[:, 5], plotStyles[0])
        plotInd += 1
        
        plt.figure(plotInd)
        plt.plot(times, vertexArray[:, 6], plotStyles[0])
        plotInd += 1
    
        plt.figure(plotInd)
        plt.plot(times, removedGraphStats[:, graphStats.numComponentsIndex], plotStyles[0])
        plotInd += 1

        plt.figure(plotInd)
        plt.plot(times2, dists, plotStyles[0])
        plotInd += 1
        meanDists.append(dists)
        print(numpy.array(dists).mean())
    
    meanDists = numpy.array(meanDists).mean(0)
    plt.plot(times2, meanDists, "b")  
    
    plt.show()
