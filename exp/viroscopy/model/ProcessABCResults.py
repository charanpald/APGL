import logging
import sys
import numpy
import scipy.stats 
import matplotlib.pyplot as pyplot
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.util.Latex import Latex
from apgl.viroscopy.model.HIVGraph import HIVGraph
from apgl.viroscopy.model.HIVRates import HIVRates
from apgl.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from apgl.viroscopy.model.HIVABCParameters import HIVABCParameters

"""
Take a set of theta values and 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

resultsDir = PathDefaults.getOutputDir() + "viroscopy/"
thetaFileName =  resultsDir + "thetaDistSimulated.pkl"

thetasArray = Util.loadPickle(thetaFileName)
meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)

precision = 3
print(Latex.array1DToRow(meanTheta, precision))
print(Latex.array1DToRow(stdTheta, precision))


#for i in range(thetasArray.shape[1]):
#    pyplot.figure()
#    pyplot.hist(thetasArray[:, i])



#A bit of code to take each theta value, run the epidemic model and save the
#number of removed
M = 1000

timesFileName = resultsDir + "epidemicTimes.pkl"
dayList = Util.loadPickle(timesFileName)
numDetectedArray = numpy.zeros((thetasArray.shape[0], len(dayList)))
T = float(dayList[-1]+1)
undirected = True
recordStep = 90
printStep = 50

for j in range(thetasArray.shape[0]):
#for j in range(5):
    theta = thetasArray[j, :]
    logging.info("theta=" + str(theta))

    graph = HIVGraph(M, undirected)
    logging.info("Created graph: " + str(graph))
    s = 3
    gen = scipy.stats.zipf(s)
    hiddenDegSeq = gen.rvs(size=graph.getNumVertices())

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates)
    model.setT(T)
    model.setRecordStep(recordStep)
    model.setPrintStep(printStep)

    params = HIVABCParameters(graph, rates)
    paramFuncs = params.getParamFuncs()

    for i in range(len(theta)):
        print(paramFuncs[i])
        if i==0:
            paramFuncs[i](int(theta[i]))
        else:
            paramFuncs[i](theta[i])

    times, infectedIndices, removedIndices, graph = model.simulate()
    numDetectedArray[j, :] = numpy.array([len(x) for x in removedIndices])

meanDetectedArray = numpy.mean(numDetectedArray, 0)
stdDetectedArray = numpy.std(numDetectedArray, 0)
print(numDetectedArray)
print(meanDetectedArray)
print(stdDetectedArray)

#Load real values
resultsFileName = resultsDir + "SimContactGrowthScalarStats.pkl"
statsArray = Util.loadPickle(resultsFileName)
graphStats = GraphStatistics()
realValues = statsArray[range(len(dayList)), graphStats.numVerticesIndex]

pyplot.errorbar(dayList, meanDetectedArray, stdDetectedArray, lolims=numpy.zeros(len(dayList)))
pyplot.hold(True)
pyplot.ylim([0, numpy.max(meanDetectedArray + stdDetectedArray) +10])
pyplot.plot(dayList, realValues, 'r')
pyplot.xlabel("Time (days)")
pyplot.ylabel("No. detected")

pyplot.show()