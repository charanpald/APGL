
import logging
import sys
import numpy
from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.sandbox.GraphMatch import GraphMatch
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
import matplotlib.pyplot as plt 

"""
This is the epidemic model for the HIV spread in cuba. We want to see how different 
graphs can get under the same params but different seeds. 
"""

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
meanTheta, sigmaTheta = HIVModelUtils.toyTheta()

epsilon = 5.0
reps = 10

graphDists = [] 
removedArray = [] 
biArray = [] 
maleArray = [] 
femaleArray = []

for i in range(reps): 
    print("i=" + str(i))
    undirected = True
    graph = HIVGraph(M, undirected)
    
    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    
    featureInds= numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
    featureInds[HIVVertices.dobIndex] = False 
    featureInds[HIVVertices.infectionTimeIndex] = False 
    featureInds[HIVVertices.hiddenDegreeIndex] = False 
    featureInds[HIVVertices.stateIndex] = False
    featureInds = numpy.arange(featureInds.shape[0])[featureInds]
    matcher = GraphMatch("PATH", alpha=0.5, featureInds=featureInds, useWeightM=False)
    graphMetrics = HIVGraphMetrics2(targetGraph, epsilon, matcher, float(endDate))
    graphMetrics.breakDist = 1.0 

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
    model.setRecordStep(recordStep)
    model.setParams(meanTheta)
    
    numpy.random.seed(i)
    times, infectedIndices, removedIndices, graph = model.simulate(True)
    print(times)
    times2, vertexArray, removedGraphStats = HIVModelUtils.generateStatistics(graph, startDate, endDate, recordStep)
    
    print(graphMetrics.dists)
    graphDists.append(graphMetrics.dists)
    removedArray.append(vertexArray[:, 0])
    maleArray.append(vertexArray[:, 1])
    femaleArray.append(vertexArray[:, 2])
    biArray.append(vertexArray[:, 4])

graphDists = numpy.array(graphDists)
removedArray = numpy.array(removedArray)
maleArray = numpy.array(maleArray)
femaleArray = numpy.array(femaleArray)
biArray = numpy.array(biArray)

graphDistsMean = graphDists.mean(0)
graphDistsStd = graphDists.std(0)

print(graphDists)
print(graphDistsMean)
print(graphDistsStd)

times = range(int(startDate), int(endDate)+1, recordStep)

plt.figure(0)
plt.errorbar(times[1:], graphDistsMean, yerr=graphDistsStd)
plt.xlabel("Time")
plt.ylabel("Objective")

plt.figure(1)
plt.errorbar(times[1:], removedArray.mean(0), yerr=removedArray.std(0), label="total")
plt.errorbar(times[1:], maleArray.mean(0), yerr=maleArray.std(0), label="male")
plt.errorbar(times[1:], femaleArray.mean(0), yerr=femaleArray.std(0), label="female")
plt.errorbar(times[1:], biArray.mean(0), yerr=biArray.std(0), label="bisexual")
plt.xlabel("Time")
plt.ylabel("No. Detected")
plt.legend(loc='upper left')
plt.show()
    