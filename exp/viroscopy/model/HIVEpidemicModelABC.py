"""
A script to estimate the HIV epidemic model parameters using ABC.
"""
from apgl.graph import *
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

#We will load up the statistics from the real graph
resultsDir = PathDefaults.getOutputDir() + "viroscopy/"
resultsFileName = resultsDir + "ContactGrowthScalarStats.pkl"
statsArray = Util.loadPickle(resultsFileName)

resultsFileName = resultsDir + "ContactGrowthVertexStats.pkl"
vertexStatsArray = Util.loadPickle(resultsFileName)

recordStep = 90
timesFileName = resultsDir + "epidemicTimes.pkl"
dayList = Util.loadPickle(timesFileName)
numTimes = len(dayList)
T = float(dayList[-1]+1)

graphStats = GraphStatistics()
numMeasures = 5
realValues = numpy.zeros((numTimes, numMeasures))
realValues[:, 0] = statsArray[list(range(len(dayList))), graphStats.numVerticesIndex]
realValues[:, 1] = statsArray[list(range(len(dayList))), graphStats.numEdgesIndex]
realValues[:, 2] = statsArray[list(range(len(dayList))), graphStats.numComponentsIndex]
realValues[:, 3] = statsArray[list(range(len(dayList))), graphStats.maxComponentSizeIndex]
realValues[:, 4] = vertexStatsArray[range(len(dayList)), 0]

logging.info("dayList=" + str(dayList))
logging.info("realValues=" + str(realValues))
diffValues = numpy.diff(realValues, axis=0)
logging.info("diffValues=" + str(diffValues))
maxDist = numpy.linalg.norm(diffValues)
logging.info("norm(diffValues)=" + str(maxDist))
epsilonArray = numpy.linspace(maxDist*0.8, maxDist*0.3, 3)
logging.info("epsilonArray=" + str(epsilonArray))

def createModel(t):
    """
    The parameter t is the particle index. 
    """
    M = 1000
    printStep = 50
    undirected = True

    graph = HIVGraph(M, undirected)
    logging.info("Created graph: " + str(graph))
    
    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())

    meanTheta = numpy.array([69, 0.5672, 2.724, 0.0029, 0.05, 0.0105, 0.1428, 0.079, 0.0784, 0.165])
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

#thetaFileName = resultsDir + "thetaReal.pkl"
#theta = Util.loadPickle(thetaFileName)
#logging.info("Real theta values: " + str(theta))

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

    
#TODO: Check the new contacts being made is realisitic range