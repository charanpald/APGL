import logging
import sys
import numpy
import scipy.stats

from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(24)

class HIVEpidemicModelProfile():
    def __init__(self):
        #Total number of people in population
        numpy.random.seed(21)                
        assert False, "Must run with -O flag"

    def profileSimulate(self):
        startDate, endDate, recordStep, printStep, M, targetGraph = HIVModelUtils.realSimulationParams()
        meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
        meanTheta = numpy.array([337,        1.4319,    0.211,     0.0048,    0.0032,    0.5229,    0.042,     0.0281,    0.0076,    0.0293])

        
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
        model.setT(startDate+100)
        model.setRecordStep(recordStep)
        model.setPrintStep(printStep)
        model.setParams(meanTheta)
        
        logging.debug("MeanTheta=" + str(meanTheta))

        ProfileUtils.profile('model.simulate()', globals(), locals())

profiler = HIVEpidemicModelProfile()
profiler.profileSimulate() #67.7
