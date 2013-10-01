"""
A script to estimate the HIV epidemic model parameters using ABC for real data.
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

import os
import logging
import sys
import numpy
import multiprocessing

assert False, "Must run with -O flag"

if len(sys.argv) > 1:
    numProcesses = int(sys.argv[1])
else: 
    numProcesses = multiprocessing.cpu_count()

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/" 
startDate, endDates, numRecordSteps, M, targetGraph = HIVModelUtils.realSimulationParams()

epsilonArray = numpy.linspace(0.4, -0.1, 6)
posteriorSampleSize = 20
breakDist = 0.7
alpha = 2
zeroVal = 0.9
matchAlpha = 0.2
logging.debug("Posterior sample size " + str(posteriorSampleSize))
logging.debug("epsilon = " + str(epsilonArray))

for i, endDate in enumerate(endDates): 
    logging.debug("="*10 + "Starting new simulation batch" + "="*10) 
    logging.debug("Total time of simulation is " + str(endDate-startDate))    
    
    def createModel(t):
        """
        The parameter t is the particle index. 
        """
        undirected = True
        graph = HIVGraph(M, undirected)
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
        
        featureInds= numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
        featureInds[HIVVertices.dobIndex] = False 
        featureInds[HIVVertices.infectionTimeIndex] = False 
        featureInds[HIVVertices.hiddenDegreeIndex] = False 
        featureInds[HIVVertices.stateIndex] = False
        featureInds = numpy.arange(featureInds.shape[0])[featureInds]
        matcher = GraphMatch("PATH", alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
        graphMetrics = HIVGraphMetrics2(targetGraph, breakDist, matcher, float(endDate))
        
        recordStep = (endDate-startDate)/float(numRecordSteps)
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
        model.setRecordStep(recordStep)
    
        return model

    pertScale = 0.05
    meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
    abcParams = HIVABCParameters(meanTheta, sigmaTheta, pertScale)
    thetaDir = resultsDir + "theta" + str(i) + "/"
    
    if not os.path.exists(thetaDir): 
        os.mkdir(thetaDir)
    
    abcSMC = ABCSMC(epsilonArray, createModel, abcParams, thetaDir, True)
    abcSMC.setPosteriorSampleSize(posteriorSampleSize)
    abcSMC.setNumProcesses(numProcesses)
    abcSMC.batchSize = 50
    abcSMC.maxRuns = 1500
    thetasArray = abcSMC.run()
    
    meanTheta = numpy.mean(thetasArray, 0)
    stdTheta = numpy.std(thetasArray, 0)
    logging.debug(thetasArray)
    logging.debug("meanTheta=" + str(meanTheta))
    logging.debug("stdTheta=" + str(stdTheta))
    
    logging.debug("New epsilon array: " + str(abcSMC.epsilonArray))
    logging.debug("Number of ABC runs: " + str(abcSMC.numRuns))

logging.debug("All done!")
