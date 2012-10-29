
import logging
import sys
import numpy
import multiprocessing 
from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.sandbox.GraphMatch import GraphMatch
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from exp.viroscopy.model.HIVABCParameters import HIVABCParameters
import matplotlib.pyplot as plt 

"""
We wish to determine the smoothness of graphs in the parameter space in terms 
of graph distance. We do this by looking at the size of the derivative 
"""

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

def findDerivative(args):
    pertScale, startDate, endDate, recordStep, M, targetGraph, seed = args
    numpy.random.seed(seed)
    meanTheta, sigmaTheta = HIVModelUtils.toyTheta()  
    
    epsilon = 5.0
    undirected = True
    
    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    
    graph = HIVGraph(M, undirected)
    
    featureInds= numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
    featureInds[HIVVertices.dobIndex] = False 
    featureInds[HIVVertices.infectionTimeIndex] = False 
    featureInds[HIVVertices.hiddenDegreeIndex] = False 
    featureInds[HIVVertices.stateIndex] = False
    featureInds = numpy.arange(featureInds.shape[0])[featureInds]
    matcher = GraphMatch("PATH", alpha=0.5, featureInds=featureInds, useWeightM=False)    
        
    abcParams = HIVABCParameters(meanTheta, sigmaTheta, pertScale)
    newTheta = abcParams.perturbationKernel(meanTheta)
    
    undirected = True
    graph = HIVGraph(M, undirected)
    graphMetrics = HIVGraphMetrics2(targetGraph, epsilon, matcher, float(endDate))
    graphMetrics.breakDist = 1.0     
    
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
    model.setRecordStep(recordStep)
    model.setParams(meanTheta)
    
    times, infectedIndices, removedIndices, graph = model.simulate(True)
    
    return abs(0.7 - graphMetrics.distance())/numpy.linalg.norm(newTheta-meanTheta)


startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()

numReps = 8
pertScales = [0.01, 0.05, 0.1, 0.2]
derivatives = numpy.zeros((numReps, len(pertScales)))

for j, pertScale in enumerate(pertScales): 
    logging.debug(pertScale)
    paramList = []
    for i in range(numReps): 
        paramList.append((pertScale, startDate, endDate, recordStep, M, targetGraph, i))
        
    pool = multiprocessing.Pool(multiprocessing.cpu_count())               
    resultIterator = pool.map(findDerivative, paramList)  
    #resultIterator = map(findDerivative, paramList)  
    pool.terminate()
    
    for i, deriv in enumerate(resultIterator): 
        derivatives[i, j] = deriv 

    
print(derivatives)
print(derivatives.mean(0))

#Seems that derivative is large for small perturbations 
#Try with other theta values 