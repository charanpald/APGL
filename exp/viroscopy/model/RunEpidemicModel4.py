
import logging
import sys
import numpy
from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from apgl.graph.GraphStatistics import GraphStatistics
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

"""
We look at the sensitivity in parameters for the epidemic model.  
"""

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)


def runModel(meanTheta): 
    startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
    endDate = 2000.0
    recordStep = 50 
    undirected = True

    logging.debug("MeanTheta=" + str(meanTheta))
    numReps = 10
    numInfectedIndices = [] 
    numRemovedIndices = []
    numRemovedEdges = []
    numContactEdges = []
    
    statistics = GraphStatistics()
    
    for i in range(numReps): 
        graph = HIVGraph(M, undirected)
        logging.info("Created graph at index " + str(i) + ": " + str(graph))
        
        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())        
        
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates)
        model.setT0(startDate)
        model.setT(endDate)
        model.setRecordStep(recordStep)
        model.setParams(meanTheta)
        times, infectedIndices, removedIndices, graph = model.simulate(True)
        
        vertexArray, contactGraphStats, removedGraphStats = HIVModelUtils.generateStatistics(graph, times)
    
        numInfectedIndices.append([len(x) for x in infectedIndices]) 
        numRemovedIndices.append([len(x) for x in removedIndices])
        
        numContactEdges.append(contactGraphStats[:, statistics.numVerticesIndex])
        numRemovedEdges.append(removedGraphStats[:, statistics.numVerticesIndex])
        
        
        
    numInfectedIndices = numpy.array(numInfectedIndices)
    numInfectedIndices = numpy.mean(numInfectedIndices, 0)
    
    numRemovedIndices = numpy.array(numRemovedIndices)
    numRemovedIndices = numpy.mean(numRemovedIndices, 0)
    
    numContactEdges = numpy.array(numContactEdges)
    numContactEdges = numpy.mean(numContactEdges, 0)
    
    numRemovedEdges = numpy.array(numRemovedEdges)
    numRemovedEdges = numpy.mean(numRemovedEdges, 0)
    
    return times, numInfectedIndices, numRemovedIndices, numContactEdges, numRemovedEdges, vertexArray[:, 6]

def plotResults(meanThetas, labels, k): 
    for i in range(meanThetas.shape[0]): 
        times, numInfectedIndices, numRemovedIndices, numContactEdges, numRemovedEdges, contactTraced = runModel(meanThetas[i, :])
        print("About to plot")

        plt.figure(k+0)
        plt.plot(times, numInfectedIndices, label=labels[i])
        plt.xlabel("Time")
        plt.ylabel("Infected")
        plt.legend()
        
        plt.figure(k+1)
        plt.plot(times, numRemovedIndices, label=labels[i])
        plt.xlabel("Time")
        plt.ylabel("Removed")
        plt.legend()
        
        plt.figure(k+2)
        plt.plot(times, contactTraced, label=labels[i])
        plt.xlabel("Time")
        plt.ylabel("Contact Traced")
        plt.legend()
   
        plt.figure(k+3)
        plt.plot(times, numContactEdges, label=labels[i])
        plt.xlabel("Time")
        plt.ylabel("Contact graph size")
        plt.legend()
     
        plt.figure(k+4)
        plt.plot(times, numRemovedEdges, label=labels[i])
        plt.xlabel("Time")
        plt.ylabel("Removed graph size")
        plt.legend()

numPlots = 5
i = 0

plotInitialInfects = False 
plotAlpha = False 
plotGamma = False 
plotBeta = True 
plotLambda = False 
plotSigma = False
 
#Number of initial infects 
if plotInitialInfects: 
    meanThetas = [[ 40,   0.9,    0.1,    0.00,    100,        0.1,      0.001]]
    meanThetas.append([ 50,   0.9,    0.1,    0.0,    100,        0.1,      0.001])
    meanThetas.append([ 60,   0.9,    0.1,    0.0,    100,        0.1,      0.001])
    meanThetas = numpy.array(meanThetas)
    labels = ["I_0=40", "I_0=50", "I_0=60"]
    plotResults(meanThetas, labels, i)
    i += numPlots


#alpha
if plotAlpha: 
    meanThetas = [[ 50,   0.6,    0.1,    0.0,    100,        0.1,      0.001]]
    meanThetas.append([ 50,   0.7,    0.1,    0.0,    100,        0.1,      0.001])
    meanThetas.append([ 50,   0.8,    0.1,    0.0,    100,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.1,    0.0,    100,        0.1,      0.001])
    meanThetas.append([ 50,   1.0,    0.1,    0.0,    100,        0.1,      0.001])
    meanThetas = numpy.array(meanThetas)
    labels = ["alpha=0.6", "alpha=0.7", "alpha=0.8", "alpha=0.9", "alpha=1.0"]
    plotResults(meanThetas, labels, i)
    i += numPlots


#gamma - random detection rate 
if plotGamma: 
    meanThetas = [[ 50,   0.9,    0.00,    0.01,    100,        0.1,      0.001]]
    meanThetas.append([ 50,   0.9,    0.05,    0.01,    100,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.10,    0.01,    100,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.20,    0.01,    100,        0.1,      0.001])
    meanThetas = numpy.array(meanThetas)
    labels = ["gamma=0.00", "gamma=0.05", "gamma=0.10", "gamma=0.2"]
    plotResults(meanThetas, labels, i)
    i += numPlots

#beta - contact detection rate 
if plotBeta: 
    meanThetas = [[ 50,   0.9,    0.01,    0.01,    1000,        0.1,      0.001]]
    meanThetas.append([ 50,   0.9,    0.01,    0.02,    1000,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.01,    0.04,    1000,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.01,    0.08,    1000,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.01,    0.16,    1000,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.01,    0.32,    1000,        0.1,      0.001])
    meanThetas = numpy.array(meanThetas)
    labels = ["beta=0.01", "beta=0.02", "beta=0.04", "beta=0.08", "beta=0.16", "beta=0.32"]
    plotResults(meanThetas, labels, i)
    i += numPlots

#lambda - contact rate 
if plotLambda: 
    meanThetas = [[ 50,   0.9,    0.1,    0.01,    100,        0.05,      0.001]]
    meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.001])
    meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.15,      0.001])
    meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.2,      0.001])
    meanThetas = numpy.array(meanThetas)
    labels = ["lambda=0.05", "lambda=0.1", "lambda=0.15", "lambda=0.2"]
    plotResults(meanThetas, labels, i)
    i += numPlots

#sigma - infection rate 
if plotSigma: 
    meanThetas = [[ 10,   0.9,    0.0,    0.0,    100,        0.1,      0.0]]
    meanThetas.append([ 10,   0.9,    0.0,    0.0,    100,        0.1,      0.001])
    meanThetas.append([ 10,   0.9,    0.0,    0.0,    100,        0.1,      0.002])
    meanThetas.append([ 10,   0.9,    0.0,    0.0,    100,        0.1,      0.004])
    meanThetas.append([ 10,   0.9,    0.0,    0.0,    100,        0.1,      0.008])
    meanThetas = numpy.array(meanThetas)
    labels = ["sigma=0.0", "sigma=0.001", "sigma=0.002", "sigma=0.004", "sigma=0.008"]
    #labels = ["sigma=0.0", "sigma=0.004"]
    plotResults(meanThetas, labels, i)
    i += numPlots

plt.show()