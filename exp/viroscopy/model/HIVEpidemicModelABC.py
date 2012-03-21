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
dataDir = PathDefaults.getOutputDir() + "viroscopy/toy" 
resultsDir = PathDefaults.getOutputDir() + "viroscopy/"
resultsFileName = resultsDir + "ContactGrowthScalarStats.pkl"

#We load a toy graph 
dataFile = dataDir + "ToyEpidemicGraph0.zip"


#Change code so that we pass into ABC a function to create a new model and 
#one to generate the parameters. 
def createModel(t):
    """
    The parameter t is the particle index. 
    """
    undirected = True
    T, recordStep, printStep, M = HIVModelUtils.defaultSimulationParams()
    graph = HIVGraph(M, undirected)
    logging.debug("Created graph: " + str(graph))
    
    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())

    meanTheta = HIVModelUtils.defaultTheta()
    rates = HIVRates(graph, hiddenDegSeq)
    abcParams = HIVABCParameters(graph, rates, meanTheta)
    
    model = HIVEpidemicModel(graph, rates)
    model.setT(T)
    model.setRecordStep(recordStep)
    model.setPrintStep(printStep)
    model.setBreakFunction(abcParams.getBreakFunc(realValues, epsilonArray[t]))

    return model, abcParams

numProcesses = multiprocessing.cpu_count()
#numProcesses = 1
posteriorSampleSize = 50
thetaLen = 11 

#Create shared variables 
thetaQueue = multiprocessing.Queue()
distQueue = multiprocessing.Queue()
summaryQueue = multiprocessing.Queue()
args = (thetaQueue, distQueue, summaryQueue)
abcList = []

for i in range(numProcesses):
    abcList.append(ABCSMC(args, epsilonArray, realValues, createModel))
    abcList[i].setPosteriorSampleSize(posteriorSampleSize)
    abcList[i].start()

logging.info("All processes started")

for i in range(numProcesses):
    abcList[i].join()

logging.info("Queue size = " + str(thetaQueue.qsize()))
thetasArray = numpy.zeros((thetaQueue.qsize(), thetaLen))

for i in range(thetaQueue.qsize()):
    thetasArray[i, :] = numpy.array(thetaQueue.get())

meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)
logging.info(thetasArray)
logging.info("meanTheta=" + str(meanTheta))
logging.info("stdTheta=" + str(stdTheta))

thetaFileName =  resultsDir + "thetaDistSimulated.pkl"
Util.savePickle(thetasArray, thetaFileName)
