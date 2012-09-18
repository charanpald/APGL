"""
Use unconstained optimisation to find the best parameters. 
"""

from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVABCParameters import HIVABCParameters
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.sandbox.GraphMatch import GraphMatch
from apgl.predictors.ABCSMC import ABCSMC, loadThetaArray

import logging
import sys
import numpy
import multiprocessing
import scipy.optimize

assert False, "Must run with -O flag"

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
logging.debug("Total time of simulation is " + str(endDate-startDate))

breakDist = 0.5

def runModel(params):
    """
    The parameter t is the particle index. 
    """
    minTheta = numpy.zeros(12)
    minTheta[6] = 1
    maxTheta = numpy.ones(12)
    maxTheta[0] = 1000
    maxTheta[6] = 1000
    params = numpy.clip(params, minTheta, maxTheta)    
    
    logging.debug("Theta = " + str(params))
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
    graphMetrics = HIVGraphMetrics2(targetGraph, breakDist, matcher, endDate)
    graphMetrics.breakDist = 0.0 

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
    model.setRecordStep(recordStep)
    model.setParams(params)
    model.simulate()

    return model.distance()

def minFunc(args): 
    (i, theta, alg, options, jacobian, tol) = args 
    result =  scipy.optimize.minimize(runModel, theta, method=alg, options=options, jac=jacobian, tol=tol)
    
    logging.debug("Accepting theta=" + str(result.x) + " dist=" + str(result.fun))
    fileName = thetaDir + "theta_i="+str(i)
    distArray = numpy.array([result.fun])            
    numpy.savez(fileName, result.x, distArray)
    return result 

if len(sys.argv) > 1:
    numProcesses = int(sys.argv[1])
else: 
    numProcesses = multiprocessing.cpu_count()

meanTheta, sigmaTheta = HIVModelUtils.toyTheta()



#Load initial guess from toy data
N = 8
t = 0  
thetaDir = resultsDir + "theta/"
thetaArray, dists = loadThetaArray(N, thetaDir, t)

paramList = []
for i in range(N): 
    args = (i, thetaArray[i, :], 'BFGS', {"maxiter":2}, False, 0.05)
    paramList.append(args)
    
pool = multiprocessing.Pool(processes=numProcesses)               
resultsIterator = pool.map(minFunc, paramList)     
#resultsIterator = map(minFunc, paramList)     
pool.terminate()    

for result in resultsIterator:     
    logging.debug(result.x)
    logging.debug(result.fun)

logging.debug("All done!")
