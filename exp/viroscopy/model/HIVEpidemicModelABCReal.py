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
from exp.viroscopy.HIVGraphReader import HIVGraphReader, CsvConverters
from exp.sandbox.GraphMatch import GraphMatch
from apgl.predictors.ABCSMC import ABCSMC

import logging
import sys
import numpy
import multiprocessing

assert False, "Must run with -O flag"

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/" 
hivReader = HIVGraphReader()
targetGraph = hivReader.readSimulationHIVGraph()

recordStep = 100 
printStep = 100 
#This needs to be from 1986 to 2004 
M = targetGraph.size * 2
startDate = CsvConverters.dateConv("01/01/1984")
endDate = CsvConverters.dateConv("01/01/1989")
#endDate = CsvConverters.dateConv("31/12/2004")

logging.debug("Total time of simulation is " + str(endDate-startDate))
epsilonArray = numpy.array([0.9, 0.6, 0.4])

def createModel(t):
    """
    The parameter t is the particle index. 
    """
    undirected = True
    M = targetGraph.size * 2
    graph = HIVGraph(M, undirected)
    
    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    
    featureInds= numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
    featureInds[HIVVertices.dobIndex] = False 
    featureInds[HIVVertices.infectionTimeIndex] = False 
    featureInds[HIVVertices.hiddenDegreeIndex] = False 
    featureInds = numpy.arange(featureInds.shape[0])[featureInds]
    matcher = GraphMatch("U", featureInds=featureInds)
    graphMetrics = HIVGraphMetrics2(targetGraph, epsilonArray[t], matcher, float(endDate))

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
thetaLen = 10

logging.debug("Posterior sample size " + str(posteriorSampleSize))

sigmaScale = 2.0 
meanTheta = HIVModelUtils.defaultTheta()
abcParams = HIVABCParameters(meanTheta, sigmaScale, 0.2)
thetaDir = resultsDir + "theta/"

abcSMC = ABCSMC(epsilonArray, createModel, abcParams, thetaDir)
abcSMC.setPosteriorSampleSize(posteriorSampleSize)
thetasArray = abcSMC.run()

meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)
logging.debug(thetasArray)
logging.debug("meanTheta=" + str(meanTheta))
logging.debug("stdTheta=" + str(stdTheta))

thetaFileName =  resultsDir + "ThetaDistReal.pkl"
Util.savePickle(thetasArray, thetaFileName)
