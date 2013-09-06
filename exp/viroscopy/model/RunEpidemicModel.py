
import logging
import sys
import numpy
from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
import matplotlib.pyplot as plt 

"""
This is the epidemic model for the HIV spread in cuba. We try to get more bisexual 
contacts  
"""

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.realSimulationParams()
endDate = startDate + 10000
meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
meanTheta = numpy.array([ 50,        0.5131,    0.3242,    0.1,    0.0001,    0.0,  325,        0.34,      0.001,     0.1])
outputDir = PathDefaults.getOutputDir() + "viroscopy/"

undirected = True
graph = HIVGraph(M, undirected)
logging.info("Created graph: " + str(graph))

alpha = 2
zeroVal = 0.9
p = Util.powerLawProbs(alpha, zeroVal)
hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())

rates = HIVRates(graph, hiddenDegSeq)
model = HIVEpidemicModel(graph, rates)
model.setT0(startDate)
model.setT(endDate)
model.setRecordStep(recordStep)
model.setParams(meanTheta)

logging.debug("MeanTheta=" + str(meanTheta))

times, infectedIndices, removedIndices, graph = model.simulate(True)

times, vertexArray, removedGraphStats = HIVModelUtils.generateStatistics(graph, startDate, endDate, recordStep)

plt.figure(0)
plt.plot(times, vertexArray[:, 0])
plt.xlabel("Time")
plt.ylabel("Removed")

plt.figure(1)
plt.plot(times, vertexArray[:, 4])
plt.xlabel("Time")
plt.ylabel("Bi Detect")

plt.figure(2)
plt.plot(times, vertexArray[:, 5])
plt.xlabel("Time")
plt.ylabel("Rand Detect")

plt.figure(3)
plt.plot(times, vertexArray[:, 6])
plt.xlabel("Time")
plt.ylabel("Contact Trace")
plt.show()