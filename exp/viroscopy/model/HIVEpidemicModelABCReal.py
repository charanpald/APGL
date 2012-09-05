"""
A script to estimate the HIV epidemic model parameters using ABC.
"""
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVABCParameters import HIVABCParameters
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from exp.sandbox.GraphMatch import GraphMatch
from apgl.predictors.ABCSMC import ABCSMC

import logging
import sys
import numpy
import multiprocessing

assert False, "Must run with -O flag"

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/" 
startDate, endDate, recordStep, printStep, M, targetGraph = HIVModelUtils.realSimulationParams()
epsilonArray = numpy.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
logging.debug("Total time of simulation is " + str(endDate-startDate))

def createModel(t):
    """
    The parameter t is the particle index. 
    """
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
    graphMetrics = HIVGraphMetrics2(targetGraph, epsilonArray[t], matcher, float(endDate))
    graphMetrics.breakDist = 0.0 

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
    model.setRecordStep(recordStep)
    model.setPrintStep(printStep)

    return model

if len(sys.argv) > 1:
    numProcesses = int(sys.argv[1])
else: 
    numProcesses = multiprocessing.cpu_count()

posteriorSampleSize = 20

logging.debug("Posterior sample size " + str(posteriorSampleSize))

purtScale = 0.1 
meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
abcParams = HIVABCParameters(meanTheta, sigmaTheta, purtScale)
thetaDir = resultsDir + "theta/"

abcSMC = ABCSMC(epsilonArray, createModel, abcParams, thetaDir)
abcSMC.setPosteriorSampleSize(posteriorSampleSize)
abcSMC.batchSize = 50
thetasArray = abcSMC.run()

meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)
logging.debug(thetasArray)
logging.debug("meanTheta=" + str(meanTheta))
logging.debug("stdTheta=" + str(stdTheta))
logging.debug("All done!")
