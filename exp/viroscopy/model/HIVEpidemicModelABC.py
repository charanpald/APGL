"""
A script to estimate the HIV epidemic model parameters using ABC.
"""
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GraphStatistics import GraphStatistics
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVABCParameters import HIVABCParameters
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVGraphMetrics import HIVGraphMetrics2
from apgl.predictors.ABCSMC import ABCSMC

import logging
import sys
import numpy
import multiprocessing
import scipy.stats 

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)
numpy.seterr(invalid='raise')

#First try the experiment on some toy data 
resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
graphFile = resultsDir + "ToyEpidemicGraph0"
targetGraph = HIVGraph.load(graphFile)

numTimeSteps = 10 
T, recordStep, printStep, M = HIVModelUtils.defaultSimulationParams()
times = numpy.linspace(0, T, numTimeSteps)
abcMetrics = HIVGraphMetrics2(times)

realSummary = abcMetrics.summary(targetGraph)
epsilonArray = numpy.array([0.5, 0.2])*numTimeSteps

def breakFunc(graph, currentTime): 
    return abcMetrics.shouldBreak(realSummary, graph, epsilonArray[0], currentTime)

def createModel(t):
    """
    The parameter t is the particle index. 
    """
    undirected = True
    T, recordStep, printStep, M = HIVModelUtils.defaultSimulationParams()
    graph = HIVGraph(M, undirected)
    
    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, T)
    model.setRecordStep(recordStep)
    model.setPrintStep(printStep)
    model.setBreakFunction(breakFunc) 

    return model

numProcesses = multiprocessing.cpu_count()
#numProcesses = 1
posteriorSampleSize = 10
thetaLen = 10

meanTheta = HIVModelUtils.defaultTheta()
abcParams = HIVABCParameters(meanTheta)

#Create shared variables 
thetaQueue = multiprocessing.Queue()
distQueue = multiprocessing.Queue()
summaryQueue = multiprocessing.Queue()
args = (thetaQueue, distQueue, summaryQueue)
abcList = []

for i in range(numProcesses):
    abcList.append(ABCSMC(args, epsilonArray, realSummary, createModel, abcParams, abcMetrics))
    abcList[i].setPosteriorSampleSize(posteriorSampleSize)
    abcList[i].start()

logging.debug("All processes started")

for i in range(numProcesses):
    abcList[i].join()

logging.debug("Queue size = " + str(thetaQueue.qsize()))
thetasArray = numpy.zeros((thetaQueue.qsize(), thetaLen))

for i in range(thetaQueue.qsize()):
    thetasArray[i, :] = numpy.array(thetaQueue.get())

meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)
logging.info(thetasArray)
logging.info("meanTheta=" + str(meanTheta))
logging.info("stdTheta=" + str(stdTheta))
logging.debug("realTheta=" + str(HIVModelUtils.defaultTheta()))

thetaFileName =  resultsDir + "ThetaDistSimulated.pkl"
Util.savePickle(thetasArray, thetaFileName)
