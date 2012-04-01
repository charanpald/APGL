
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
from exp.viroscopy.model.HIVGraphMetrics import HIVGraphMetrics2
from exp.sandbox.GraphMatch import GraphMatch 

"""
This is the epidemic model for the HIV spread in cuba. We repeat the simulation a number
of times and average the results. The purpose is to test the ABC model selection 
by using a known value of theta. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

T, recordStep, printStep, M = HIVModelUtils.defaultSimulationParams()

numRepetitions = 10
undirected = True
outputDir = PathDefaults.getOutputDir() + "viroscopy/toy/"
theta = HIVModelUtils.defaultTheta() 

thetaFileName = outputDir + "ThetaToy.pkl"
Util.savePickle(theta, thetaFileName)

graphList = []

for j in range(numRepetitions):
    graph = HIVGraph(M, undirected)
    logging.debug("Created graph: " + str(graph))

    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates)
    model.setT(T)
    model.setRecordStep(recordStep)
    model.setPrintStep(printStep)
    model.setParams(theta)
    
    logging.debug("Theta = " + str(theta))
    
    times, infectedIndices, removedIndices, graph = model.simulate(True)
    graphFileName = outputDir + "ToyEpidemicGraph" + str(j)
    graph.save(graphFileName)
    
    graphList.append(graph)

logging.debug("All done. Computing graph distances ... ")
