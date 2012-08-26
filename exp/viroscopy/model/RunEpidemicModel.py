
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
This is the epidemic model for the HIV spread in cuba. We try to get more bisexual 
contacts  
"""

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

startDate, endDate, recordStep, printStep, M, targetGraph = HIVModelUtils.realSimulationParams()
meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
meanTheta = numpy.array([143,        0.5185,    0.4358,    0.0919,    0.0013,   0.0942,   0.2381,    0.0189,    0.0189,    0.0187,    0.0192])
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



removedInds = list(graph.getRemovedSet())
removedGraph = graph.subgraph(removedInds)

print((removedGraph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.bi).sum())
print((removedGraph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.hetero).sum())

#Vary initial infects 
