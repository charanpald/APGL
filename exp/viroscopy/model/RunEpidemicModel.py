
import logging
import sys
import numpy
from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVABCParameters import HIVABCParameters
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2

"""
This is the epidemic model for the HIV spread in cuba. We repeat the simulation a number
of times and average the results. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

startDate, endDate, recordStep, printStep, M, targetGraph = HIVModelUtils.realSimulationParams()
meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
meanTheta = numpy.array([337,        1.4319,    0.211,     0.0048,    0.0032,    0.5229,    0.042,     0.0281,    0.0076,    0.0293])
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
model.setPrintStep(printStep)
model.setParams(meanTheta)

logging.debug("MeanTheta=" + str(meanTheta))

times, infectedIndices, removedIndices, graph = model.simulate(True)


