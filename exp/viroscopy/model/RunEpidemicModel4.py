
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
    recordStep = 50 
    undirected = True

    logging.debug("MeanTheta=" + str(meanTheta))
    numReps = 5
    numInfectedIndices = [] 
    numRemovedIndices = []
    numEdges = []
    
    statistics = GraphStatistics()
    
    for i in range(numReps): 
        graph = HIVGraph(M, undirected)
        logging.info("Created graph: " + str(graph))
        
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
        
        vertexArray, removedGraphStats = HIVModelUtils.generateStatistics(graph, times)
    
        numInfectedIndices.append([len(x) for x in infectedIndices]) 
        numRemovedIndices.append([len(x) for x in removedIndices])
        numEdges.append(removedGraphStats[:, statistics.numEdgesIndex])
        
    numInfectedIndices = numpy.array(numInfectedIndices)
    numInfectedIndices = numpy.mean(numInfectedIndices, 0)
    
    numRemovedIndices = numpy.array(numRemovedIndices)
    numRemovedIndices = numpy.mean(numRemovedIndices, 0)
    
    numEdges = numpy.array(numEdges)
    numEdges = numpy.mean(numEdges, 0)
    
    return times, numInfectedIndices, numRemovedIndices, numEdges

def plotResults(meanThetas, labels, k): 
    for i in range(meanThetas.shape[0]): 
        times, numInfectedIndices, numRemovedIndices, numEdges = runModel(meanThetas[i, :])

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
        plt.plot(times, numEdges, label=labels[i])
        plt.xlabel("Time")
        plt.ylabel("Edges")
        plt.legend()

numPlots = 3
 
#Number of initial infects 
i = 0
meanThetas = [[ 40,   0.9,    0.1,    0.00,    100,        0.1,      0.001]]
meanThetas.append([ 50,   0.9,    0.1,    0.0,    100,        0.1,      0.001])
meanThetas.append([ 60,   0.9,    0.1,    0.0,    100,        0.1,      0.001])
meanThetas = numpy.array(meanThetas)
labels = ["I_0=40", "I_0=50", "I_0=60"]
plotResults(meanThetas, labels, i)


#alpha
i += numPlots
meanThetas = [[ 50,   0.6,    0.1,    0.0,    100,        0.1,      0.001]]
meanThetas.append([ 50,   0.7,    0.1,    0.0,    100,        0.1,      0.001])
meanThetas.append([ 50,   0.8,    0.1,    0.0,    100,        0.1,      0.001])
meanThetas.append([ 50,   0.9,    0.1,    0.0,    100,        0.1,      0.001])
meanThetas.append([ 50,   1.0,    0.1,    0.0,    100,        0.1,      0.001])
meanThetas = numpy.array(meanThetas)
labels = ["alpha=0.6", "alpha=0.7", "alpha=0.8", "alpha=0.9", "alpha=1.0"]
plotResults(meanThetas, labels, i)
plt.show()

#gamma - random detection rate 
i += numPlots
meanThetas = [[ 50,   0.9,    0.05,    0.01,    100,        0.1,      0.001]]
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.001])
meanThetas.append([ 50,   0.9,    0.15,    0.01,    100,        0.1,      0.001])
meanThetas.append([ 50,   0.9,    0.2,    0.01,    100,        0.1,      0.001])
meanThetas = numpy.array(meanThetas)
labels = ["gamma=0.05", "gamma=0.1", "gamma=0.15", "gamma=0.2"]
plotResults(meanThetas, labels, i)

#beta - contact detection rate 
i += numPlots
meanThetas = [[ 50,   0.9,    0.1,    0.005,    100,        0.1,      0.001]]
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.001])
meanThetas.append([ 50,   0.9,    0.1,    0.015,    100,        0.1,      0.001])
meanThetas.append([ 50,   0.9,    0.1,    0.02,    100,        0.1,      0.001])
meanThetas = numpy.array(meanThetas)
labels = ["beta=0.005", "beta=0.01", "beta=0.015", "beta=0.02"]
plotResults(meanThetas, labels, i)

#lambda - contact rate 
i += numPlots
meanThetas = [[ 50,   0.9,    0.1,    0.01,    100,        0.05,      0.001]]
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.001])
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.15,      0.001])
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.2,      0.001])
meanThetas = numpy.array(meanThetas)
labels = ["lambda=0.05", "lambda=0.1", "lambda=0.15", "lambda=0.2"]
plotResults(meanThetas, labels, i)

#sigma - infection rate 
i += numPlots
meanThetas = [[ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.001]]
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.002])
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.004])
meanThetas.append([ 50,   0.9,    0.1,    0.01,    100,        0.1,      0.008])
meanThetas = numpy.array(meanThetas)
labels = ["lambda=0.001", "lambda=0.002", "lambda=0.004", "lambda=0.008"]
plotResults(meanThetas, labels, i)

plt.show()